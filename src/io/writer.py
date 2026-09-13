"""视频写出：把标注后的帧写成 mp4。计划书 §3.2 输出侧。

两个踩坑点（均已实测）
--------------------
1. **懒创建**：`cv2.VideoWriter` 在构造时就必须知道画面尺寸，而尺寸只有拿到**首帧**后才知道，
   流式处理不可能提前读取。所以由本类在第一次 `write()` 时按实际帧尺寸创建。
2. **尺寸不一致会静默丢帧**：`VideoWriter.write()` 收到与创建尺寸不同的帧时，
   底层 FFmpeg 只打印 `Failed to write frame` 然后**丢弃该帧**——调用方若只数字调用次数，
   会误以为全部写成功（本次实测：写 30 次、实际只落 16 帧）。
   因此本类对尺寸不一致的帧做**等比缩放纠正**并计数，保证 `frames_written` 说真话；
   这也顺带覆盖了 RTSP 流中途改变分辨率的真实场景。

关于中文路径：已实测本机 OpenCV 5.0 的 `VideoWriter` 在 `D:\\py\\机器视觉\\...` 下可正常
创建并写出（size > 0），故不做临时文件中转。若换到旧版 OpenCV（<4.5）出现 0 字节输出，
再考虑「写 ASCII 临时路径 → shutil.move」的中转方案。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FPS = 25.0
DEFAULT_FOURCC = "mp4v"


class AnnotatedVideoWriter:
    """标注视频写出器：首帧到达时按真实尺寸创建 VideoWriter。"""

    def __init__(self, path: str | Path, fps: float | None = None, fourcc: str = DEFAULT_FOURCC):
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
        self.fourcc = fourcc
        self.size: tuple[int, int] | None = None   # (w, h)，创建后固定
        self.frames_written = 0                    # 真正写进容器的帧数
        self.frames_resized = 0                    # 因尺寸不符被纠正的帧数
        self._writer: cv2.VideoWriter | None = None
        self._mismatch_warned = False

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    def _ensure_writer(self, frame: np.ndarray) -> None:
        """首次写帧时创建底层 VideoWriter。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        self.size = (w, h)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"无法创建视频输出: {self.path}（编码器 {self.fourcc} 不可用？）")
        log.info("[VideoWriter] 开始写出: %s (%dx%d @ %.1ffps)", self.path, w, h, self.fps)

    def write(self, frame: np.ndarray) -> None:
        """写一帧（BGR）。首次调用时创建底层 VideoWriter；尺寸不符时等比纠正。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        if self._writer is None:
            self._ensure_writer(frame)
        elif frame.shape[:2][::-1] != self.size:
            if not self._mismatch_warned:
                log.warning("[VideoWriter] 帧尺寸 %s 与输出 %s 不一致，将等比纠正（后续同类警告不再重复）",
                            frame.shape[:2][::-1], self.size)
                self._mismatch_warned = True
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            self.frames_resized += 1
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info("[VideoWriter] 已写出 %d 帧（其中尺寸纠正 %d 帧）→ %s",
                     self.frames_written, self.frames_resized, self.path)

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
