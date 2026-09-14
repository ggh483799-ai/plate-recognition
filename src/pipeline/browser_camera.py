"""浏览器摄像头会话：帧由**浏览者自己的电脑**推上来，服务端只做识别。

问题背景（为什么要单独一条链路）
------------------------------
页面部署到云服务器后，原来的「本机摄像头」调 `/stream/devices` —— 那是**服务器**上的
摄像头（`cv2.VideoCapture(index)` 打开的是服务进程所在主机上的设备）。云服务器没接摄像头，
用户于是看到"未检测到可用摄像头"。这不是设备问题，是**语义错位**：

    浏览器摄像头（本模块）  摄像头在"看网页的人"手里 → 服务端打不开它，只能由浏览器推帧
    服务器摄像头            摄像头插在服务器上（页面里作为流地址填 0/1 仍可用）
    流地址 / 网页链接        RTSP / 视频网页链接，服务端主动拉（LiveStreamSession）

架构取舍（为什么不回传画面）
--------------------------
客户端本地用 `<video>` 原生播放（30fps 流畅、零延迟），服务端只回**识别结果**
（车牌号 + 坐标 + 统计），前端在视频上叠一层 canvas 画框。好处：

- 上行一帧 JPEG（约 30-60 KB）+ 下行一小段 JSON，带宽是"服务端推 MJPEG"的几分之一；
- 画面流畅度与识别速度彻底解耦（识别 0.5 秒一次，画面照样 30fps）；
- 服务端不再有编码/推流线程，CPU 只花在推理上。

识别节奏与"只认最新帧"
--------------------
- 客户端按 `interval_ms` 推帧；服务端**再次**按同一间隔节流（客户端时钟不可信），
  间隔内到达的帧直接沿用上次结果（`reused=true`，成本≈0）；
- 推理用**非阻塞锁**：上一帧还在算时，新来的帧直接丢弃并计数（`recog_dropped`）——
  与拉流会话的"队列容量 1"语义一致：过期结果没有价值。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.stream import (
    FINAL_STATUSES,
    STATUS_RUNNING,
    STATUS_STARTING,
    STATUS_STOPPED,
    stats_payload,
)
from src.pipeline.video_pipeline import PlateDeduplicator, ShotWriter
from src.vision.draw import crop_plate

log = logging.getLogger(__name__)

# 浏览器推帧的 backend 标识（前端统计面板会显示）
BACKEND_CLIENT_PUSH = "浏览器推帧"


class ClientCameraSession:
    """一个"浏览器摄像头"会话：吃帧 → 识别 → 回结果（不回画面）。"""

    def __init__(
        self,
        session_id: str,
        pipeline,
        *,
        name: str = "浏览器摄像头",
        out_dir: str | Path | None = None,
        interval_ms: int = 200,
        max_side: int = 640,
        dedup_window_s: float = 3.0,
        save_shots: bool = True,
        show_objects: bool = True,
        idle_timeout_s: float = 45.0,
        media_base_prefix: str = "/stream-media",
    ):
        self.session_id = session_id
        self.name = name
        self.pipeline = pipeline
        self.media_base_prefix = media_base_prefix.rstrip("/")
        self.interval_s = max(0.0, float(interval_ms) / 1000.0)
        self.max_side = int(max_side)
        self.show_objects = bool(show_objects)
        self.idle_timeout_s = float(idle_timeout_s)
        self.live = True                      # 语义上就是直播（摄像头）
        self.out_dir = Path(out_dir) if out_dir else None

        self.dedup = PlateDeduplicator(dedup_window_s)
        self.shots = ShotWriter(self.out_dir / "shots") if (self.out_dir is not None and save_shots) else None

        self.status = STATUS_STARTING
        self.error = ""
        self.frames = 0              # 服务端接收并解码成功的帧数（≈上行帧率）
        self.reused = 0              # 因节流/占用而直接沿用上次结果的帧数
        self.recog_runs = 0
        self.recog_dropped = 0       # 推理忙时被丢弃的帧数
        self.recog_cost_ms = 0.0
        self.frame_size: tuple[int, int] = (0, 0)
        self.started_at = time.time()
        self.finished_at = 0.0
        self.last_frame_ts = 0.0     # 最后一次收到帧的时间（空闲回收依据）

        self._lock = threading.Lock()
        self._run_lock = threading.Lock()      # 非阻塞：同一时刻只允许一次推理
        self._last_results: list = []
        self._last_vehicles: list = []
        self._last_persons: list = []
        self._last_rejected: list = []
        self._last_run_ts = 0.0
        self._event_seq = 0
        self._event_seqs: dict[str, int] = {}

    # ========================================================
    # 生命周期
    # ========================================================

    @property
    def is_finished(self) -> bool:
        return self.status in FINAL_STATUSES

    def stop(self) -> None:
        """收尾（幂等）。摄像头数据流在客户端手里，停掉本地采集由前端负责。"""
        with self._lock:
            if self.status not in FINAL_STATUSES:
                self.status = STATUS_STOPPED
            self.finished_at = self.finished_at or time.time()
        log.info("[Camera] %s 结束: frames=%d recog=%d reuse=%d dropped=%d events=%d",
                 self.session_id, self.frames, self.recog_runs, self.reused,
                 self.recog_dropped, len(self.dedup.history))

    def idle_seconds(self) -> float:
        """距上次收到帧的秒数（用于回收"用户关掉页面"的会话）。"""
        if not self.last_frame_ts:
            return time.time() - self.started_at
        return time.time() - self.last_frame_ts

    # ========================================================
    # 核心：吃一帧，回结果
    # ========================================================

    def ingest(self, data: bytes) -> dict:
        """处理客户端推来的一帧 JPEG，返回识别结果 + 统计（不含画面）。

        返回的 `plates` / `objects` 的 bbox 坐标都在**这一帧的像素空间**里，
        `frame_size` 给出该空间尺寸——前端据此换算到 `<video>` 的显示尺寸。
        """
        frame = self._resize(self._decode(data))
        now = time.time()
        with self._lock:
            self.frames += 1
            self.last_frame_ts = now
            if self.status == STATUS_STARTING:
                self.status = STATUS_RUNNING
            due = (now - self._last_run_ts) >= self.interval_s

        # 节流：间隔内的帧不重复推理，直接沿用上次结果（成本≈0，客户端也不必等）
        if not due and self.recog_runs:
            with self._lock:
                self.reused += 1
            return self._payload(reused=True)

        # 只认最新帧：上一帧还在算就丢掉这一帧（不排队、不攒延迟）
        if not self._run_lock.acquire(blocking=False):
            with self._lock:
                self.recog_dropped += 1
                self.reused += 1
            return self._payload(reused=True)
        try:
            t0 = time.perf_counter()
            results = self.pipeline.run(frame)
            cost_ms = (time.perf_counter() - t0) * 1000
        finally:
            self._run_lock.release()

        idx = self.frames
        t_sec = now - self.started_at
        with self._lock:
            self._last_run_ts = now
            self.recog_runs += 1
            self.recog_cost_ms += cost_ms
            self._last_results = results
            self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
            self._last_vehicles = list(getattr(self.pipeline, "last_vehicles", []) or [])
            self._last_persons = list(getattr(self.pipeline, "last_persons", []) or [])
            self._last_rejected = list(getattr(self.pipeline, "last_rejected", []) or [])
            fresh = self.dedup.accept(results, idx, t_sec)
            for ev in fresh:
                self._event_seq += 1
                self._event_seqs[ev.plate_no] = self._event_seq
            updates = self.dedup.pop_best_updates() if self.shots is not None else []
        if updates:
            self._save_shots(frame, updates)
        log.debug("[Camera] %s 第 %d 帧: 车牌 %d 个, 识别 %.0fms",
                  self.session_id, idx, len(results), cost_ms)
        return self._payload(reused=False)

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        """JPEG/PNG 字节 → BGR 图。解码失败抛 ValueError（调用方回 400）。"""
        if not data:
            raise ValueError("收到空帧")
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("帧不是有效的图片数据（需要 JPEG/PNG）")
        return frame

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """按 max_side 降采样：上传尺寸由客户端决定，但服务端不信任它。"""
        h, w = frame.shape[:2]
        longest = max(h, w)
        if self.max_side and longest > self.max_side:
            scale = self.max_side / longest
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        return frame

    def _save_shots(self, frame: np.ndarray, events: list) -> None:
        for ev in events:
            crop = crop_plate(frame, ev.bbox)
            if crop is None:
                continue
            seq = self._event_seqs.get(ev.plate_no, 0)
            ev.shot_img = crop
            self.shots.save(ev, name=f"shot_{seq:03d}.jpg" if seq else None)

    # ========================================================
    # 对外读取
    # ========================================================

    def _objects(self) -> list[dict]:
        """侧信道里的车辆/行人/疑似车牌候选（供前端画细线框）。

        注意 `pipeline.last_vehicles` 的元素是 `(BBox, 名称)` 二元组——
        置信度在 BBox 上（`v.score`），不是三元组。
        """
        if not self.show_objects:
            return []
        out: list[dict] = []
        for bbox, name in self._last_vehicles:
            out.append({"kind": "vehicle", "label": str(name),
                        "score": round(float(getattr(bbox, "score", 0) or 0), 3),
                        "bbox": [float(v) for v in bbox.xyxy]})
        for p in self._last_persons:
            out.append({"kind": "person", "label": "person",
                        "score": round(float(getattr(p, "score", 0) or 0), 3),
                        "bbox": [float(v) for v in p.xyxy]})
        for r, reason in self._last_rejected:
            out.append({"kind": "candidate", "label": str(reason),
                        "score": round(float(r.det_score), 3),
                        "bbox": [float(v) for v in r.bbox]})
        return out

    def _payload(self, *, reused: bool) -> dict:
        with self._lock:
            plates = [
                {
                    "plate_no": r.plate_no,
                    "plate_color": r.plate_color,
                    "vehicle_type": r.vehicle_type,
                    "det_score": round(float(r.det_score), 4),
                    "rec_score": round(float(r.rec_score), 4),
                    "bbox": [float(v) for v in r.bbox],
                }
                for r in self._last_results
            ]
            return {
                "session_id": self.session_id,
                "status": self.status,
                "reused": reused,
                "plates": plates,
                "objects": self._objects(),
                "frame_size": [self.frame_size[0], self.frame_size[1]],
                "events": [ev.to_dict() for ev in self.dedup.history],
                "media_base": self.media_base,
                "stats": self._stats(),
            }

    @property
    def media_base(self) -> str:
        """截图目录的对外地址。前缀由服务层传入——两套会话的产物目录是分开挂载的。"""
        return f"{self.media_base_prefix}/{self.session_id}/shots" if self.shots is not None else ""

    def _stats(self) -> dict:
        elapsed = max(1e-6, (self.finished_at or time.time()) - self.started_at)
        return stats_payload(
            frames=self.frames,
            read_frames=self.frames,
            skipped=self.reused,
            recog_runs=self.recog_runs,
            recog_dropped=self.recog_dropped,
            recog_cost_ms=self.recog_cost_ms,
            events=len(self.dedup.history),
            total_hits=self.dedup.total_hits,
            elapsed=elapsed,
            fps=None,
            total_frames=0,
            live=True,
            interval_s=self.interval_s,
            max_side=self.max_side,
            jpeg_quality=0,               # 不回传画面：没有服务端编码
            backend=BACKEND_CLIENT_PUSH,
        )

    def snapshot(self) -> dict:
        """与 `LiveStreamSession.snapshot()` 同结构（前端与接口模型可直接复用）。"""
        with self._lock:
            stats = self._stats()
            status, error = self.status, self.error
            events = [ev.to_dict() for ev in self.dedup.history]
        return {
            "session_id": self.session_id,
            "status": status,
            "source": self.name,
            "error": error,
            "stats": stats,
            "source_title": self.name,
            "events": events,
            "media_base": self.media_base,
        }
