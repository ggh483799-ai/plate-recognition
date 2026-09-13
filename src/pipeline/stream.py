"""实时流识别：一边播放视频一边识别（逐帧读 → 异步识别 → 分色画框 → MJPEG 推流）。

为什么必须单独一条路径（不能复用 `VideoPipeline`）
------------------------------------------------
`VideoPipeline.process()` 是"把整段读干、最后汇总落盘"的**离线**模型，实时场景要的正好相反：

1. **边读边出**：每读到一帧就编码推送，客户端立刻看到画面；
2. **识别必须与播放解耦**：CPU 上单次识别约 0.3–1.3 秒（实测 640px + low 档约 1.29s ≈ 0.8 次/秒）。
   若在同一循环里同步识别，画面就会被识别卡住——实测 32 帧的视频 5 秒只推了 4 帧，
   看起来像幻灯片。因此识别放在**独立线程**，队列容量 1（旧帧直接覆盖丢弃）：
   画面按源帧率走，识别结果按算力节奏刷新，中间帧沿用上一次的框。
   **这才是"一边播放视频一边识别"**：播放流畅度与识别速度互不拖累；
3. **按源帧率节流读取**（文件源）→ 真的在"播放"视频，而不是尽可能快地跑完；
4. **直播源（摄像头/RTSP）落后时丢帧追赶** → 保住"实时性"，宁可丢帧也不要越播越延迟。

推流格式选 MJPEG
---------------
`multipart/x-mixed-replace` + JPEG，浏览器 `<img src>` 原生支持：前端不需要 JS 解码、
也不需要 WebSocket。代价是没法回补丢掉的帧——但对"实时预览"来说"永远显示最新帧"才是对的。

产物
----
`out_dir/shots/shot_NNN.jpg`：**事件首次出现就写**，之后每次刷新到更好的帧就**覆盖同名文件**
（与离线不同——离线是定稿才写一次）。这样页面能边播边看到证据图。

线程模型
--------
    [读帧线程 _run]     读帧 → 按源帧率节流 → 提交识别(非阻塞) → 画框 → JPEG → 发布最新帧
    [识别线程 _recognize_loop]  取最新帧 → pipeline.run → 去重/事件/截图 → 更新"当前结果"
    [HTTP 消费端 _mjpeg_frames]  wait_frame 取最新帧推给浏览器（多客户端各自持有游标）
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.io.online import (
    ffmpeg_capture_options,
    is_webpage_link,
    resolve_online,
    download_online,
)
from src.io.reader import iter_frames, make_reader
from src.pipeline.video_pipeline import PlateDeduplicator, ShotWriter
from src.vision.draw import (
    CANDIDATE_COLOR,
    crop_plate,
    draw_detections,
    draw_objects,
    load_font,
    object_color,
)

log = logging.getLogger(__name__)

STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

FINAL_STATUSES = (STATUS_STOPPED, STATUS_FAILED)

# 「直播源」前缀：帧由对端按自己的节奏推过来，本地不按帧率节流，落后时丢帧追赶。
# 注意 http(s):// 不算直播源——它多半是一个远端视频文件，按文件对待（可按 fps 节流）。
LIVE_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "udp://", "tcp://")

# 识别线程等待新帧/结束信号的时间上限（便于 stop() 及时收尾）
RECOG_GET_TIMEOUT = 0.5


def is_live_source(source) -> bool:
    """是否直播源（摄像头索引 / RTSP-RTMP-UDP 流）。"""
    text = str(source).strip().lower()
    if text.isdigit():
        return True
    return text.startswith(LIVE_PREFIXES)


class LiveStreamSession:
    """一路实时识别会话：读帧线程 + 识别线程 + 最新帧缓冲，可随时 stop，状态可查。"""

    def __init__(
        self,
        session_id: str,
        source,
        pipeline,
        *,
        name: str = "",
        out_dir: str | Path | None = None,
        interval_ms: int = 0,
        max_side: int = 640,
        jpeg_quality: int = 75,
        display_fps: float = 10.0,
        dedup_window_s: float = 3.0,
        idle_timeout_s: float = 60.0,
        first_frame_timeout_s: float = 30.0,
        save_shots: bool = True,
        show_objects: bool = True,
    ):
        self.session_id = session_id
        self.source = source
        self.name = name or str(source)
        self.pipeline = pipeline
        self.interval_s = max(0.0, float(interval_ms) / 1000.0)
        self.max_side = int(max_side)
        self.jpeg_quality = int(jpeg_quality)
        self.display_interval = 1.0 / max(1.0, float(display_fps))
        self.idle_timeout_s = float(idle_timeout_s)
        self.first_frame_timeout_s = float(first_frame_timeout_s)
        self.first_frame_ms = 0.0   # 首帧到达耗时（设备慢/被占用时能一眼看出来）
        self.backend = ""           # 采集后端名（摄像头才有）
        self.media_title = ""       # 网页链接解析出的视频标题（微博/B站等）
        self.show_objects = bool(show_objects)  # 是否在画面上标注车辆/行人与"疑似车牌"
        self.live = is_live_source(source)
        self.out_dir = Path(out_dir) if out_dir else None

        self.dedup = PlateDeduplicator(dedup_window_s)
        # 没有 out_dir 就不落图（单测里常见），省掉磁盘副作用
        self.shots = ShotWriter(self.out_dir / "shots") if (self.out_dir is not None and save_shots) else None

        # ---- 状态（读写都在 self._lock 下）----
        self.status = STATUS_STARTING
        self.error = ""
        self.frames = 0            # 已推送给客户端的帧数
        self.read_frames = 0       # 已从源读到的帧数（含未推送的）
        self.skipped = 0           # 为追赶实时性而丢弃的帧数
        self.recog_runs = 0
        self.recog_dropped = 0     # 因识别队列满而直接丢掉帧（说明算力跟不上，是设计如此）
        self.recog_cost_ms = 0.0
        self.warmup_ms = 0.0       # 启动阶段预热的耗时（把它从"首次识别"里挪出来）
        self.fps: float | None = None
        self.total_frames = 0
        self.started_at = time.time()
        self.finished_at = 0.0
        self._last_results: list = []
        self._last_vehicles: list = []    # [(BBox, 名称)] —— 识别线程写入，推流线程读取
        self._last_persons: list = []
        self._last_rejected: list = []    # [(PlateResult, 原因)]
        self._event_seq = 0
        self._event_seqs: dict[str, int] = {}   # 车牌号 → 最新事件序号（决定小图文件名）

        # ---- 线程与发布 ----
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._recog_q: queue.Queue = queue.Queue(maxsize=1)
        self._latest: bytes | None = None
        self._seq = 0
        self._closed = False                    # 收尾完成 → 消费端据此结束流
        self._last_pull_ts = time.time()
        self._thread: threading.Thread | None = None
        self._recog_thread: threading.Thread | None = None

    # ========================================================
    # 生命周期
    # ========================================================

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("会话已启动，不能重复 start()")
        self._recog_thread = threading.Thread(
            target=self._recognize_loop, name=f"recog-{self.session_id}", daemon=True)
        self._recog_thread.start()
        self._thread = threading.Thread(
            target=self._run, name=f"stream-{self.session_id}", daemon=True)
        self._thread.start()
        # 首帧看门狗单独一线程：读帧线程可能永久阻塞在 cap.read() 上，自身无法自救
        threading.Thread(target=self._watchdog, name=f"watch-{self.session_id}", daemon=True).start()

    def stop(self, timeout: float = 3.0) -> None:
        """请求停止并等待工作线程收尾。幂等，可重复调用。"""
        self._stop.set()
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._stop_recognizer()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                # 读 RTSP 时 cap.read() 可能长时间阻塞，join 超时属正常，不是错误
                log.warning("[Stream] %s 读帧线程未在 %.1fs 内退出（可能阻塞在读取上）",
                            self.session_id, timeout)

    @property
    def is_finished(self) -> bool:
        return self.status in FINAL_STATUSES

    def _watchdog(self) -> None:
        """首帧看门狗：设备"打开成功但永不吐帧"时，不能让会话假装在跑。

        摄像头 / RTSP 地址不通、需要认证、或虚拟设备只返回空流时，`cap.read()` 会**一直阻塞**，
        会话状态停在 running、画面永远空白——用户只会看到"没反应"。这里给出可读的失败原因，
        并主动收尾把 MJPEG 流关掉（否则浏览器端也一直挂着）。
        """
        if self.first_frame_timeout_s <= 0:
            return
        # 先等进入 running（要跳过模型预热的 starting 阶段），再开始计时
        wait_until = time.time() + self.first_frame_timeout_s + 60.0
        status = STATUS_STARTING
        while time.time() < wait_until:
            with self._lock:
                status = self.status
            if status != STATUS_STARTING:
                break
            time.sleep(0.2)
        if status != STATUS_RUNNING:
            return

        deadline = time.time() + self.first_frame_timeout_s
        while time.time() < deadline:
            with self._lock:
                if self.read_frames > 0 or self.status != STATUS_RUNNING:
                    return
            time.sleep(0.5)

        with self._lock:
            if self.read_frames > 0 or self.status != STATUS_RUNNING:
                return
            self.status = STATUS_FAILED
            self.error = (f"打开成功但 {self.first_frame_timeout_s:.0f}s 内没有读到任何帧"
                          "（设备被其它程序占用 / 驱动未就绪 / 地址不可达 / 需要认证？）")
            message = self.error
        log.error("[Stream] %s %s", self.session_id, message)
        self.stop()      # 读帧线程可能仍阻塞在 read 上，由 join 超时兜底

    # ========================================================
    # 对外读取
    # ========================================================

    def wait_frame(self, last_seq: int = -1, timeout: float = 5.0) -> tuple[int, bytes | None]:
        """等待一帧比 `last_seq` 更新的 JPEG，返回 `(新序号, JPEG字节)`。

        流已结束且没有新帧时返回 `(last_seq, None)`，调用方据此结束推流。
        **多客户端安全**：每个客户端自己持有 `last_seq`，不会互相"偷帧"。

        注意判断条件是「序号变了**且确实有帧**」——只看序号会在会话刚启动（序号 0、还没
        推过任何帧）时误判为"有新帧"并返回 `None`，让客户端以为流已经结束。
        """
        with self._cond:
            self._last_pull_ts = time.time()

            def has_new() -> bool:
                return self._seq != last_seq and self._latest is not None

            if not has_new() and not self._closed:
                self._cond.wait_for(lambda: has_new() or self._closed, timeout)
            if not has_new():
                return last_seq, None       # 收尾了，或这次等超时——两种都由调用方决定后续
            return self._seq, self._latest

    def snapshot(self) -> dict:
        """一份状态快照（可在运行中安全调用）。"""
        with self._lock:
            elapsed = max(1e-6, (self.finished_at or time.time()) - self.started_at)
            events = [ev.to_dict() for ev in self.dedup.history]
            stats = {
                "frames": self.frames,
                "read_frames": self.read_frames,
                "skipped": self.skipped,
                "recog_runs": self.recog_runs,
                "recog_dropped": self.recog_dropped,
                "avg_recog_ms": round(self.recog_cost_ms / self.recog_runs, 1) if self.recog_runs else 0.0,
                "events": len(self.dedup.history),
                "total_hits": self.dedup.total_hits,
                "elapsed_s": round(elapsed, 1),
                "push_fps": round(self.frames / elapsed, 1),
                "recog_fps": round(self.recog_runs / elapsed, 2),
                "source_fps": self.fps,
                "source_frames": self.total_frames or None,
                "live": self.live,
                "interval_ms": int(round(self.interval_s * 1000)),
                "max_side": self.max_side,
                "jpeg_quality": self.jpeg_quality,
                "warmup_ms": round(self.warmup_ms, 1),
                "first_frame_ms": round(self.first_frame_ms, 1),
                "backend": self.backend,
            }
            status, error = self.status, self.error
        return {
            "session_id": self.session_id,
            "status": status,
            "source": self.name,
            "error": error,
            "stats": stats,
            "source_title": self.media_title,
            "events": events,
            "media_base": f"/stream-media/{self.session_id}/shots" if self.shots is not None else "",
        }

    def _resolve_source(self):
        """把会话来源变成可拉流的地址。

        网页视频链接（微博 / 抖音 / B站 / YouTube …）不是流地址——OpenCV 打开网页拿到的是
        HTML，实测"打开成功但永远读不到帧"。必须先经 yt-dlp 解析出**媒体直链**再拉流；
        CDN 需要的 Referer/UA 头一并透传。解析失败时如实置 failed 并给出可读原因，
        不回退原链接空转 30s 等看门狗（用户只会更困惑）。

        直链也拉不动时（实测 B站 upos CDN 对 FFmpeg 透传头不认账，仍 403），
        降级为**下载到本地再播放**——VOD 场景下载完才能开播，属如实取舍。
        """
        src = self.source
        if not (isinstance(src, str) and is_webpage_link(src)):
            return src

        media = resolve_online(src, max_height=self._height_cap())
        opts = ffmpeg_capture_options(media.headers)
        if opts:
            # 该环境变量在 cv2.VideoCapture 构造时读取。多会话并发时值基本一致
            # （都是常规 UA/Referer），竞态影响可忽略；单会话场景完全确定。
            import os

            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
        with self._lock:
            self.media_title = media.title or src

        if self._probe_playable(media.play_url):
            return media.play_url

        # 直链拉不动 → 下载兜底
        import tempfile

        base = self.out_dir if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="lpr-dl-"))
        local = download_online(media, Path(base) / "source.mp4")
        log.info("[Stream] %s 直链不可拉流，已下载兜底: %s", self.session_id, local.name)
        return str(local)

    @staticmethod
    def _probe_playable(play_url: str) -> bool:
        """直链能否被 OpenCV 拉流（打开且读到一帧才算数，isOpened 会骗人）。"""
        try:
            cap = cv2.VideoCapture(play_url)
            try:
                if not cap.isOpened():
                    return False
                ok, _ = cap.read()
                return bool(ok)
            finally:
                cap.release()
        except Exception:  # noqa: BLE001 —— 探测绝不能把会话搞挂
            return False

    def _height_cap(self) -> int:
        """按推理画面宽度选视频清晰度：拉 1080p 再缩到 320px 纯属浪费。"""
        from src.io.online import height_cap_for

        return height_cap_for(self.max_side)

    def _warmup(self) -> float:
        """跑一次空白帧，把"首次推理"的初始化开销挪到启动阶段。

        失败不影响会话：只是首次识别会慢一些，所以只记 WARNING。
        """
        t0 = time.perf_counter()
        try:
            self.pipeline.run(np.zeros((360, 640, 3), dtype=np.uint8))
        except Exception as exc:  # noqa: BLE001
            log.warning("[Stream] %s 预热失败（不影响使用，首次识别会偏慢）: %s", self.session_id, exc)
        return (time.perf_counter() - t0) * 1000

    # ========================================================
    # 读帧线程
    # ========================================================

    def _run(self) -> None:
        t0 = time.perf_counter()
        deadline = t0
        last_submit = 0.0
        last_push = 0.0
        idx = -1
        first_frame = True
        reader = None
        try:
            reader = make_reader(self._resolve_source(), self.max_side)
            fps = getattr(reader, "fps", None)
            with self._lock:
                self.fps = float(fps) if fps else None
                self.total_frames = int(getattr(reader, "frame_count", 0) or 0)
                self.backend = getattr(reader, "backend", "") or ""

            # 预热：首次推理要初始化 ultralytics 的 predictor，实测首次识别要 3s+。
            # 挪到启动阶段做完，用户看到的第一辆车就不会先等 3 秒（代价是"开始"慢一点）。
            warmup_ms = self._warmup()
            with self._lock:
                self.warmup_ms = warmup_ms
                self.status = STATUS_RUNNING
            log.info("[Stream] %s 启动: source=%s live=%s fps=%s interval=%dms size=%d 后端=%s 预热=%.0fms",
                     self.session_id, self.name, self.live, self.fps,
                     int(round(self.interval_s * 1000)), self.max_side,
                     self.backend or "-", warmup_ms)

            for frame in iter_frames(reader):
                if self._stop.is_set():
                    break
                idx += 1
                now = time.perf_counter()
                if first_frame:
                    # 首帧才建立播放节奏原点，并记录"拿到首帧用了多久"。
                    # 设备初始化/驱动唤醒耗掉的时间**不能**算成"播放落后"，否则会触发一次巨型追赶：
                    # 实测摄像头首帧慢 15s 时一次性 grab 掉 444 帧，等于把内容整段跳过去了。
                    first_frame = False
                    deadline = now
                    with self._lock:
                        self.first_frame_ms = (now - t0) * 1000
                    log.info("[Stream] %s 首帧就绪: %.0fms（后端 %s）",
                             self.session_id, self.first_frame_ms, self.backend or "-")

                # --- 1) 播放节奏：按源帧率走；直播源落后太多就丢帧追赶 ---
                if self.fps:
                    deadline += 1.0 / self.fps
                    lag = deadline - time.perf_counter()
                    if lag > 0:
                        time.sleep(min(lag, 0.2))       # 上限 0.2s，保证 stop() 及时生效
                    elif self.live and lag < -2.0 / self.fps:
                        deadline = self._catch_up(reader, deadline)

                # --- 2) 提交识别（非阻塞：队列满就丢，永远只认最新帧）---
                if now - last_submit >= self.interval_s:
                    if self._submit_recog(frame, idx, self._t_sec(idx, t0)):
                        last_submit = now

                # --- 3) 推送：限流到 display_fps，省 CPU 与带宽 ---
                # 注意必须用**此刻**的时间：上面按源帧率 sleep 过，若沿用循环开头取的 now，
                # 差值恒为 0，会导致每隔一帧才推一次（实测 8fps 的源只推成 2.9fps）。
                t_push = time.perf_counter()
                if t_push - last_push >= self.display_interval:
                    last_push = t_push
                    self._push(frame, idx, t0)

                with self._lock:
                    self.read_frames += 1

                # --- 4) 没人看就自动停（浏览器关了标签页 / 客户端断开）---
                if (self.idle_timeout_s > 0 and idx % 30 == 0
                        and time.time() - self._last_pull_ts > self.idle_timeout_s):
                    log.info("[Stream] %s 超过 %.0fs 无客户端消费，自动停止",
                             self.session_id, self.idle_timeout_s)
                    break
        except Exception as exc:  # 源打不开 / 解码异常 → 状态置 failed，不抛出到线程外
            log.exception("[Stream] %s 处理失败", self.session_id)
            with self._lock:
                self.status = STATUS_FAILED
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._stop_recognizer()
            with self._lock:
                if self.status != STATUS_FAILED:
                    self.status = STATUS_STOPPED
                self.finished_at = time.time()
                snap = (self.status, self.frames, self.recog_runs, len(self.dedup.history))
            with self._cond:
                self._closed = True
                self._cond.notify_all()
            log.info("[Stream] %s 结束: status=%s frames=%d recog=%d events=%d",
                     self.session_id, snap[0], snap[1], snap[2], snap[3])

    # ========================================================
    # 识别线程
    # ========================================================

    def _recognize_loop(self) -> None:
        """只处理"最新一帧"：队列容量 1，识别期间涌进来的帧直接被丢弃覆盖。

        这既是性能选择（永远不欠账、不排队攒延迟），也是产品选择：实时场景里
        "晚来的旧结果"没有价值，**当前画面的结果**才有价值。
        """
        while True:
            try:
                item = self._recog_q.get(timeout=RECOG_GET_TIMEOUT)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if item is None:          # 结束信号
                break
            frame, idx, t_sec = item
            try:
                t0 = time.perf_counter()
                results = self.pipeline.run(frame)
                cost_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:  # 单帧识别失败不该杀掉整条流
                log.warning("[Stream] %s 识别失败（跳过该帧）: %s", self.session_id, exc)
                continue

            with self._lock:
                fresh = self.dedup.accept(results, idx, t_sec)
                for ev in fresh:
                    self._event_seq += 1
                    self._event_seqs[ev.plate_no] = self._event_seq
                updates = self.dedup.pop_best_updates() if self.shots is not None else []
                self.recog_runs += 1
                self.recog_cost_ms += cost_ms
                self._last_results = results
                # 诊断侧信道：车辆/行人与"疑似但没过阈值"的车牌候选，供推流线程画框
                self._last_vehicles = list(getattr(self.pipeline, "last_vehicles", []) or [])
                self._last_persons = list(getattr(self.pipeline, "last_persons", []) or [])
                self._last_rejected = list(getattr(self.pipeline, "last_rejected", []) or [])
            if updates:
                self._save_shots(frame, updates)

    def _submit_recog(self, frame: np.ndarray, idx: int, t_sec: float) -> bool:
        """把一帧交给识别线程；队列满说明上一帧还没算完，直接丢弃（返回 False）。"""
        try:
            self._recog_q.put_nowait((frame, idx, t_sec))
            return True
        except queue.Full:
            with self._lock:
                self.recog_dropped += 1
            return False

    def _stop_recognizer(self, timeout: float = 5.0) -> None:
        """给识别线程发结束信号并等它收尾（幂等）。"""
        q = self._recog_q
        try:
            q.put_nowait(None)
        except queue.Full:
            # 队列里还压着一帧：先腾一个位置再塞结束信号，否则线程会一直等下去
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(None)
            except queue.Full:
                log.warning("[Stream] %s 无法投递结束信号，识别线程将随超时退出", self.session_id)
        thread = self._recog_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                log.warning("[Stream] %s 识别线程未在 %.1fs 内退出（可能仍在推理）",
                            self.session_id, timeout)

    # ========================================================
    # 内部
    # ========================================================

    def _t_sec(self, idx: int, t0: float) -> float:
        """时间轴。文件源用帧号/帧率（可复现）；直播源用挂钟（丢帧后帧号已不代表时间）。"""
        if self.fps and not self.live:
            return idx / self.fps
        return time.perf_counter() - t0

    def _catch_up(self, reader, deadline: float) -> float:
        """直播源落后时丢弃若干帧，返回新的 deadline（保住低延迟）。

        **单次追赶有上限**（约 2 秒的帧量）：否则一次长时间停顿（设备唤醒、网络抖动、断流重连）
        会让这个循环一口气丢几百帧、把整段内容跳过去——实测摄像头首帧慢 15s 时丢了 444 帧。
        剩下没追上的部分交给后续循环慢慢消化。
        """
        skip = getattr(reader, "skip", None)
        if not callable(skip) or not self.fps:
            return deadline
        max_skip = max(1, int(self.fps * 2))
        dropped = 0
        while dropped < max_skip and not self._stop.is_set():
            lag = deadline - time.perf_counter()
            if lag >= -1.0 / self.fps:
                break
            if skip(1) <= 0:
                break
            deadline += 1.0 / self.fps
            dropped += 1
        if dropped:
            with self._lock:
                self.skipped += dropped
        return deadline

    def _save_shots(self, frame: np.ndarray, events: list) -> None:
        """把"刷新了最佳值"的事件小图落盘（同名覆盖，始终保留当前最好的一张）。"""
        for ev in events:
            crop = crop_plate(frame, ev.bbox)
            if crop is None:
                continue
            seq = self._event_seqs.get(ev.plate_no, 0)
            ev.shot_img = crop
            self.shots.save(ev, name=f"shot_{seq:03d}.jpg" if seq else None)

    def _push(self, frame: np.ndarray, idx: int, t0: float) -> None:
        """画框 → JPEG → 发布为"最新帧"。

        三层标注，刻意区分开：
            车牌（粗实线 + 号牌，识别结果）＞ 车辆/行人（细线 + 类别）＞ 疑似车牌（灰细线 + 原因）。
        第三层是给用户看的"为什么没识别出这块牌"：检测到了，只是置信度没过阈值。
        """
        with self._lock:
            results = list(self._last_results)
            vehicles = list(self._last_vehicles)
            persons = list(self._last_persons)
            rejected = list(self._last_rejected)
            frames, recog = self.frames, self.recog_runs
        elapsed = max(1e-6, time.perf_counter() - t0)
        hud = [
            f"frame {idx}",
            f"push {frames / elapsed:.1f}fps",
            f"recog {recog / elapsed:.2f}/s",
            f"plate {len(results)}",
        ]
        if self.show_objects and (vehicles or persons):
            hud.append(f"obj {len(vehicles) + len(persons)}")

        canvas = draw_detections(frame, results, hud=hud)
        if self.show_objects:
            objects = [(v.xyxy, f"{name} {float(getattr(v, 'score', 0)):.2f}", object_color(name))
                       for v, name in vehicles]
            objects += [(p.xyxy, f"person {float(getattr(p, 'score', 0)):.2f}", object_color("person"))
                        for p in persons]
            objects += [(r.bbox, f"疑似车牌 {reason}", CANDIDATE_COLOR) for r, reason in rejected]
            if objects:
                # 物体标签字号比车牌小一号，避免盖住号牌
                px = int(np.clip(canvas.shape[1] / 80, 11, 20))
                draw_objects(canvas, objects, font=load_font(px), inplace=True)

        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            log.warning("[Stream] JPEG 编码失败，跳过该帧")
            return
        jpeg = buf.tobytes()
        with self._lock:
            self.frames += 1
            self._latest = jpeg
            with self._cond:
                self._seq += 1
                self._cond.notify_all()
