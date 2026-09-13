"""视频 / 摄像头 / RTSP 流推理编排：逐帧识别 → 跨帧去重 → 结构化落盘 + 标注视频。

计划书 §3.1 数据流 + §7 难点：
- **去重不是优化项，是必需项**：同一辆车在流里会连续出现几十上百帧，逐帧上报会让下游
  （卡口、告警、车流统计）收到几十条重复记录。本模块按「车牌号 + 时间窗口」做**事件级**去重。
- 本模块只做**编排与统计**：识别算法全部在 `LprPipeline` 内，此处不碰算法，便于单测替换。

事件定稿（何时落盘）
------------------
事件**不在首帧就落盘**，而是等它「定稿」后才写：车牌离开画面（窗口内不再出现）或整个流结束。
原因：首帧写出的记录里 `hits=1`、`vehicle_type` 还是单帧的偶发误判（实测同一辆车首帧被判
`truck`、多帧综合后被纠正为 `car`、识别置信度也从 0.9353 升到 0.9998）。先写半成品再补写，
下游拿到的是自相矛盾的两份数据。代价是**上报延迟 = 去重窗口**（默认 3s），对卡口/车流统计可接受；
若后续需要更低延迟，可加 `--emit-on-first-sight` 走"先报后更正"的两段式。

最佳帧截图
----------
去重时保留的是**识别置信度最高那一帧**的框，本模块顺手把那一帧的车牌小图裁下来存到
`shot_img`（内存），定稿落盘时由 `ShotWriter` 写成 jpg。页面/报告可以直接展示证据图。
注意必须 `copy()`：`cv2.VideoCapture` 会复用帧缓冲区，不拷贝会拿到下一帧的内容。

与单图入口的关系
---------------
`src/pipeline/lpr_pipeline.py` 处理单帧；本模块在其上叠「多帧循环 + 去重 + 落盘 + 画框 + 进度」。

CLI
---
    # 视频文件
    python -m src.pipeline.video_pipeline --source clip.mp4 --out annotated.mp4 --csv detections.csv
    # 摄像头（本机 0 号）
    python -m src.pipeline.video_pipeline --source 0 --max-frames 300
    # RTSP 流
    python -m src.pipeline.video_pipeline --source rtsp://user:pwd@ip:554/stream --jsonl events.jsonl
    # 单图也能走（与 `lpr_pipeline` 同源，方便统一验证）
    python -m src.pipeline.video_pipeline --source data/field/test_car.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.common.interfaces import PlateResult
from src.io.reader import iter_frames, make_reader
from src.io.writer import AnnotatedVideoWriter
from src.vision.draw import crop_plate, draw_detections

log = logging.getLogger(__name__)

CSV_FIELDS = (
    "frame_idx",
    "t_sec",
    "timestamp",
    "plate_no",
    "plate_color",
    "vehicle_type",
    "det_score",
    "rec_score",
    "hits",
    "x1",
    "y1",
    "x2",
    "y2",
    "shot",
)

# 进度回调签名：(已处理帧数, 总帧数(0=未知), 含车牌帧数)
ProgressCB = Callable[[int, int, int], None]


# ============================================================
# 事件模型
# ============================================================

@dataclass
class PlateEvent:
    """一次「车牌出现」事件（去重后的最小上报单元）。"""

    plate_no: str = ""
    plate_color: str = ""
    vehicle_type: str = ""
    det_score: float = 0.0
    rec_score: float = 0.0
    bbox: list[float] = field(default_factory=list)
    frame_idx: int = 0          # 首次出现的帧号
    t_sec: float = 0.0          # 首次出现的时间轴秒数
    timestamp: str = ""         # 首次出现的挂钟时间（ISO）
    hits: int = 1               # 该事件累计命中的帧数
    last_t_sec: float = 0.0     # 最近一次命中的时间轴秒数
    shot: str = ""              # 最佳帧车牌小图的文件名（落盘后回填；未落盘为空）
    # 内存中的最佳帧小图。不属于上报字段：to_dict()/to_row() 都不含它，
    # 仅供 ShotWriter 落盘使用，故 repr/compare 都关掉，避免日志里刷出一大坨数组。
    shot_img: Any = field(default=None, repr=False, compare=False)

    def to_row(self) -> dict:
        """展平成一行（CSV 用）。"""
        box = list(self.bbox) + [0.0] * 4
        return {
            "frame_idx": self.frame_idx,
            "t_sec": round(float(self.t_sec), 3),
            "timestamp": self.timestamp,
            "plate_no": self.plate_no,
            "plate_color": self.plate_color,
            "vehicle_type": self.vehicle_type,
            "det_score": round(float(self.det_score), 4),
            "rec_score": round(float(self.rec_score), 4),
            "hits": self.hits,
            "x1": round(float(box[0]), 1),
            "y1": round(float(box[1]), 1),
            "x2": round(float(box[2]), 1),
            "y2": round(float(box[3]), 1),
            "shot": self.shot,
        }

    def to_dict(self) -> dict:
        return {
            "plate_no": self.plate_no,
            "plate_color": self.plate_color,
            "vehicle_type": self.vehicle_type,
            "det_score": round(float(self.det_score), 4),
            "rec_score": round(float(self.rec_score), 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "frame_idx": self.frame_idx,
            "t_sec": round(float(self.t_sec), 3),
            "timestamp": self.timestamp,
            "hits": self.hits,
            "shot": self.shot,
        }


class PlateDeduplicator:
    """跨帧去重：同一车牌在 window 秒内只算一个事件。

    语义（滑窗 + 离开判定）
    ----------------------
    - 该车牌上一次命中距当前 ≤ window 秒 → 同一事件，只累加 `hits` 并保留**最高识别置信度**
      （同一车牌在不同帧的清晰度不同，取最好的一次上报，比取最后一次更合理）；
    - 距上次命中 > window 秒 → 认为车已离开，再次出现算**新事件**（例如同一辆车绕一圈回来）。

    只依赖 `PlateResult` 的标准字段，纯逻辑，可脱离模型单测。
    """

    def __init__(self, window_s: float = 3.0):
        if window_s < 0:
            raise ValueError(f"window_s 不能为负: {window_s}")
        self.window_s = float(window_s)
        self._active: dict[str, PlateEvent] = {}
        self._closed: list[PlateEvent] = []   # 已定稿、待上报的事件
        self._best_updated: list[PlateEvent] = []  # 本帧产生了「更优帧」的事件（供截图用）
        self.history: list[PlateEvent] = []
        self.total_hits = 0

    def accept(
        self,
        results: list[PlateResult],
        frame_idx: int,
        t_sec: float,
        now: datetime | None = None,
    ) -> list[PlateEvent]:
        """喂入一帧结果，返回本帧**新开启**的事件（首现 / 离开后重现）。

        注意：新开启的事件**尚未定稿**——它的 `hits`、最佳置信度还会被后续帧更新。
        要拿到定稿事件请用 `pop_closed()`（离开时定稿）或 `flush()`（流结束时全部定稿）。
        """
        self._expire(t_sec)
        fresh: list[PlateEvent] = []
        for res in results:
            plate_no = (res.plate_no or "").strip()
            if not plate_no:
                log.debug("[Dedup] 空车牌号，跳过（帧 %d）", frame_idx)
                continue
            self.total_hits += 1

            ev = self._active.get(plate_no)
            if ev is not None and (t_sec - ev.last_t_sec) <= self.window_s:
                ev.hits += 1
                ev.last_t_sec = t_sec
                if res.rec_score > ev.rec_score:  # 取最清晰的一次作为上报值
                    ev.det_score = res.det_score
                    ev.rec_score = res.rec_score
                    ev.bbox = list(res.bbox)
                    ev.plate_color = res.plate_color
                    ev.vehicle_type = res.vehicle_type
                    self._best_updated.append(ev)  # 本帧是最佳帧 → 调用方可以裁图留证
                continue

            ev = PlateEvent(
                plate_no=plate_no,
                plate_color=res.plate_color,
                vehicle_type=res.vehicle_type,
                det_score=res.det_score,
                rec_score=res.rec_score,
                bbox=list(res.bbox),
                frame_idx=frame_idx,
                t_sec=t_sec,
                timestamp=(now or datetime.now()).isoformat(timespec="seconds"),
                hits=1,
                last_t_sec=t_sec,
            )
            self._active[plate_no] = ev
            self.history.append(ev)
            self._best_updated.append(ev)  # 首帧即为当前最佳帧
            fresh.append(ev)
        return fresh

    @property
    def events(self) -> list[PlateEvent]:
        """已产生的全部事件（按出现时间排序，含仍在场的）。"""
        return list(self.history)

    def pop_best_updates(self) -> list[PlateEvent]:
        """取出并清空「本帧产生了更优帧」的事件，供调用方裁图。

        与 `pop_closed()` 的区别：这里返回的是**刚刷新最佳值**的事件（可能还在场），
        调用方拿到后应立刻用**当前帧**按 `ev.bbox` 裁图——此刻的 bbox 就是这一帧的。
        """
        updated, self._best_updated = self._best_updated, []
        return updated

    def _expire(self, t_sec: float) -> None:
        """把超过窗口未再出现的车牌判为「已离开」，其事件定稿。"""
        gone = [no for no, ev in self._active.items() if t_sec - ev.last_t_sec > self.window_s]
        for no in gone:
            log.debug("[Dedup] 车牌 %s 已离开（%.2fs 未再出现），事件定稿", no, t_sec - self._active[no].last_t_sec)
            self._closed.append(self._active.pop(no))

    def pop_closed(self) -> list[PlateEvent]:
        """取出并清空「已定稿」事件（离开的车牌 + 已 flush 的在场事件）。"""
        closed, self._closed = self._closed, []
        return closed

    def flush(self) -> list[PlateEvent]:
        """流结束：把所有仍在场的事件一并定稿并返回（清空缓冲区）。"""
        self._closed.extend(self._active.values())
        self._active.clear()
        return self.pop_closed()

    @property
    def active_count(self) -> int:
        return len(self._active)

    def summary(self) -> dict:
        by_color: dict[str, int] = {}
        for ev in self.history:
            by_color[ev.plate_color or "unknown"] = by_color.get(ev.plate_color or "unknown", 0) + 1
        return {
            "events": len(self.history),
            "total_hits": self.total_hits,
            "distinct_plates": len({ev.plate_no for ev in self.history}),
            "by_plate_color": by_color,
            "window_s": self.window_s,
        }


# ============================================================
# 结构化落盘
# ============================================================

class CsvEventSink:
    """事件 CSV 落盘（增量写 + 每行 flush，长流中途中断也不丢已产出记录）。

    编码用 `utf-8-sig`：带 BOM，Excel 双击打开中文车牌号不乱码。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(CSV_FIELDS))
        self._writer.writeheader()
        self._fh.flush()

    def write(self, event: PlateEvent) -> None:
        self._writer.writerow(event.to_row())
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CsvEventSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class JsonlEventSink:
    """事件 JSONL 落盘（一行一条，便于下游流式消费）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, event: PlateEvent) -> None:
        self._fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlEventSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ShotWriter:
    """把事件的最佳帧车牌小图落盘，供页面 / 报告展示证据图。

    文件名用**序号**（`shot_001.jpg`）而不是车牌号：车牌含中文，中文文件名在 URL、
    zip 打包、跨系统拷贝上都是雷点，序号 + JSON 里的 `plate_no` 已经能一一对应。
    """

    def __init__(self, dir_path: str | Path, suffix: str = ".jpg"):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.suffix = suffix
        self.saved = 0        # 实际写出成功的次数
        self._auto = 0        # 自动命名计数器（显式传 name 时不消耗）

    def save(self, event: PlateEvent, name: str | None = None) -> str:
        """写出一个事件的小图，回填 `event.shot` 并返回文件名；无图时返回空串。

        `name` 显式指定文件名时，**同名会直接覆盖**——实时流（`src/pipeline/stream.py`）
        靠它做到"同一个车牌始终保留当前最好的一张"，而不是每刷新一次就多出一个文件。
        不传 `name` 就按 `shot_001.jpg` 递增命名（离线流水线用）。
        """
        img = getattr(event, "shot_img", None)
        if img is None or getattr(img, "size", 0) == 0:
            return ""
        from src.io.reader import imwrite_bgr  # 延迟导入，避免 io↔pipeline 顶层循环

        if name is None:
            self._auto += 1
            name = f"shot_{self._auto:03d}{self.suffix}"
        if not imwrite_bgr(self.dir / name, img):
            log.warning("[Shot] 小图写出失败: %s", self.dir / name)
            return ""
        self.saved += 1
        event.shot = name
        event.shot_img = None  # 及时释放，长流下避免累计占用内存
        return name


def _emit(sinks: list, events: list[PlateEvent], shots: ShotWriter | None = None) -> int:
    """把事件写入所有落盘器，返回本次写入的事件条数（每个事件在最前面 sink 里算 1 条）。"""
    for ev in events:
        if shots is not None:
            shots.save(ev)
        for sink in sinks:
            sink.write(ev)
    return len(events)


# ============================================================
# 编排
# ============================================================

class VideoPipeline:
    """多帧编排：逐帧识别 → 去重 → 落盘/画框 → 汇总统计。"""

    def __init__(self, pipeline, window_s: float = 3.0):
        self.pipeline = pipeline
        self.dedup = PlateDeduplicator(window_s)

    def process(
        self,
        reader,
        sinks: list | None = None,
        writer: AnnotatedVideoWriter | None = None,
        max_frames: int = 0,
        log_every: int = 30,
        show: bool = False,
        shots: ShotWriter | None = None,
        progress_cb: ProgressCB | None = None,
    ) -> dict:
        """处理整个来源，返回统计字典。

        参数
        ----
        reader      : FrameReader（`src.io.reader.make_reader` 产出）
        sinks       : 事件落盘器列表（每个提供 `write(event)`）
        writer      : 标注视频写出器（可选）
        max_frames  : >0 时最多处理这么多帧（RTSP 长流调试 / 服务端限流用）
        log_every   : 每 N 帧打一条进度日志
        show        : 实时预览窗口（交互式，默认关闭；ESC 退出）
        shots       : 最佳帧小图落盘器（可选；给了才会裁图，省掉无用的内存拷贝）
        progress_cb : 每帧回调 `(已处理帧数, 总帧数, 含车牌帧数)`，服务端据此上报进度
        """
        sinks = list(sinks or [])
        fps = getattr(reader, "fps", None)
        total_frames = int(getattr(reader, "frame_count", 0) or 0)
        t_start = time.perf_counter()
        t_origin: float | None = None   # 无帧率来源（图片/未知流）的时间轴原点，首帧才起算
        frames = 0
        det_frames = 0
        new_events = 0
        reported = 0
        initialized = False

        try:
            for idx, frame in enumerate(iter_frames(reader)):
                if max_frames and frames >= max_frames:
                    log.info("[Video] 达到 max_frames=%d，提前结束", max_frames)
                    break

                initialized = True   # 首帧已到手 → 开始按帧回调进度（首帧前是打开/预热阶段）
                results = self.pipeline.run(frame)
                if fps:
                    t_sec = idx / fps          # 有帧率 → 用视频时间轴（可复现、与真实时长一致）
                else:
                    # 无帧率（单图 / 未知流）→ 用挂钟，但原点是**首帧处理完**，
                    # 否则模型预热耗时会被算进第 0 帧的时间戳（实测单图 t_sec=3.18s）。
                    if t_origin is None:
                        t_origin = time.perf_counter()
                    t_sec = time.perf_counter() - t_origin

                new_events += len(self.dedup.accept(results, idx, t_sec))
                # 裁图必须在 accept 之后、且用**本帧**：此刻 ev.bbox 正是本帧最优框
                if shots is not None:
                    self._capture_shots(frame)
                # 只有「定稿」的事件才落盘：此时 hits 与最佳置信度都已完整，
                # 避免写出 hits=1、vehicle_type 还是首帧误判值的半成品记录。
                reported += _emit(sinks, self.dedup.pop_closed(), shots)
                if results:
                    det_frames += 1

                if writer is not None:
                    hud = [
                        f"frame {idx}",
                        f"plate {len(results)}",
                        f"event {len(self.dedup.history)}",
                    ]
                    writer.write(draw_detections(frame, results, hud=hud))

                frames += 1
                if progress_cb is not None:
                    progress_cb(frames, total_frames, det_frames)
                if log_every and frames % log_every == 0:
                    log.info("[Video] 已处理 %d 帧（含车牌 %d 帧，事件 %d 个）",
                             frames, det_frames, len(self.dedup.history))

                if show and not _preview(frame, results):
                    log.info("[Video] 用户按 ESC 退出预览")
                    break
        finally:
            # 流结束/异常退出：把仍在场的事件一并定稿落盘，然后才关文件
            reported += _emit(sinks, self.dedup.flush(), shots)
            for sink in sinks:
                sink.close()
            if writer is not None:
                writer.close()

        elapsed = time.perf_counter() - t_start
        # 容器声称的帧数 > 实际处理帧数 → 说明被 max_frames 截断或解码提前结束。
        # 如实上报，避免用户以为"整个视频都看过了"。
        truncated = bool(total_frames and frames < total_frames)
        stats = {
            "frames": frames,
            "total_frames": total_frames or None,
            "truncated": truncated,
            "initialized": initialized,
            "frames_with_plate": det_frames,
            "new_events": new_events,
            "events_reported": reported,
            "shots_saved": shots.saved if shots is not None else 0,
            "elapsed_s": round(elapsed, 3),
            "process_fps": round(frames / elapsed, 2) if elapsed > 0 else 0.0,
            "source_fps": round(fps, 2) if fps else None,
            "out_video": str(getattr(writer, "path", "")) if writer is not None else "",
            "video_frames_written": getattr(writer, "frames_written", 0) if writer is not None else 0,
            **self.dedup.summary(),
        }
        log.info("[Video] 处理完成: %d 帧 / %.2fs / %.1f fps / 事件 %d 个",
                 stats["frames"], stats["elapsed_s"], stats["process_fps"], stats["events"])
        return stats

    def _capture_shots(self, frame: np.ndarray) -> None:
        """为本帧刷新了最佳值的事件裁下车牌小图（`crop_plate` 内部已 copy）。"""
        for ev in self.dedup.pop_best_updates():
            crop = crop_plate(frame, ev.bbox)
            if crop is not None:
                ev.shot_img = crop


def _preview(frame: np.ndarray, results) -> bool:
    """实时预览（需图形环境）。返回 False 表示用户要求退出。"""
    import cv2

    try:
        cv2.imshow("plate-recognition", frame)
        return (cv2.waitKey(1) & 0xFF) != 27
    except cv2.error as exc:  # headless / 无 GUI 后端
        log.warning("[Video] 预览不可用，已忽略 --show: %s", exc)
        return True


# ============================================================
# CLI
# ============================================================

def main() -> int:
    """CLI 入口：python -m src.pipeline.video_pipeline --source clip.mp4"""
    from src.common.logger import setup_logger

    setup_logger("video_pipeline")

    parser = argparse.ArgumentParser(
        description="车牌识别（视频 / 摄像头 / RTSP 流），输出事件 CSV/JSONL 与标注视频"
    )
    parser.add_argument("--source", required=True,
                        help="视频文件 / 图片 / 摄像头索引(0) / RTSP URL")
    parser.add_argument("--out", default="", help="标注视频输出路径（.mp4）")
    parser.add_argument("--csv", default="", help="事件 CSV 输出路径")
    parser.add_argument("--jsonl", default="", help="事件 JSONL 输出路径")
    parser.add_argument("--shots-dir", default="", help="最佳帧车牌小图输出目录（每事件一张）")
    parser.add_argument("--window", type=float, default=3.0,
                        help="去重时间窗（秒），同车牌窗内只上报一次，默认 3.0")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数，0=不限")
    parser.add_argument("--log-every", type=int, default=30, help="每 N 帧打一条进度日志")
    parser.add_argument("--device", default="cpu", help="cpu / 0(GPU)，仅影响自研级联")
    parser.add_argument("--engine-mode", choices=("auto", "cascade", "engine"), default=None)
    parser.add_argument("--show", action="store_true", help="实时预览窗口（ESC 退出）")
    args = parser.parse_args()

    if args.device != "cpu":
        from src.common.config import load_config
        load_config("lprnet")["device"] = args.device

    from src.pipeline.lpr_pipeline import build_pipeline

    pipeline = build_pipeline()
    if args.engine_mode is not None:
        pipeline.engine_mode = args.engine_mode if pipeline.engine else "cascade"

    try:
        reader = make_reader(args.source)
    except RuntimeError as exc:
        log.error("[Video] 打开来源失败: %s", exc)
        return 1

    sinks = []
    if args.csv:
        sinks.append(CsvEventSink(args.csv))
    if args.jsonl:
        sinks.append(JsonlEventSink(args.jsonl))
    writer = AnnotatedVideoWriter(args.out, fps=getattr(reader, "fps", None)) if args.out else None
    shots = ShotWriter(args.shots_dir) if args.shots_dir else None

    vp = VideoPipeline(pipeline, window_s=args.window)
    stats = vp.process(
        reader, sinks=sinks, writer=writer, shots=shots,
        max_frames=args.max_frames, log_every=args.log_every, show=args.show,
    )

    output = {
        "code": 0,
        "msg": "success",
        "data": {
            **stats,
            "source": str(args.source),
            "plates": [ev.to_dict() for ev in vp.dedup.events],
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
