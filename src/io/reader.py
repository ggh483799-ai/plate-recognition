"""图片 / 视频 / 摄像头 / RTSP 统一读取，输出 BGR 帧。计划书 §3.2：不关心业务。

所有读取器都满足 `src.common.interfaces.FrameReader` 协议（`read()` / `close()`），
这样上层（视频流水线）无需区分来源类型——图片、mp4、摄像头、RTSP 走同一条循环。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

# 长边归一化尺寸（计划书 §3.1）
DEFAULT_MAX_SIDE = 1280

# 视为「静态图片」的后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Windows 上摄像头索引的候选后端（按实测可靠性排序）。
# 实测（同一台机器、同一个摄像头 0）：
#   DSHOW : 打开 358ms / 首帧 143ms
#   默认  : 独立进程里 打开 777ms / 首帧 617ms，但**在服务进程内首帧要 15s+**（驱动初始化卡住）
# → 索引源优先用 DSHOW，打不开再依次回退，避免"打开成功却半天不出画面"。
CAMERA_BACKENDS = ("DSHOW", "MSMF", None)


def _backend_label(backend) -> str:
    """后端常量 → 可读名字（日志与接口里都要看得懂）。"""
    if backend is None:
        return "default"
    return {
        getattr(cv2, "CAP_DSHOW", -1): "DSHOW",
        getattr(cv2, "CAP_MSMF", -2): "MSMF",
        getattr(cv2, "CAP_FFMPEG", -3): "FFMPEG",
    }.get(backend, str(backend))


def _camera_backend_list() -> list:
    out: list = []
    for name in CAMERA_BACKENDS:
        if name is None:
            out.append(None)
        elif hasattr(cv2, f"CAP_{name}"):
            out.append(getattr(cv2, f"CAP_{name}"))
    return out


def imread_bgr(path: str | Path) -> np.ndarray | None:
    """读取图片为 BGR 数组；读不到返回 None。

    **不用 cv2.imread**：它在 Windows 上走 ANSI 文件接口，路径含中文时会**静默返回 None**
    （本项目根目录 `D:\\py\\机器视觉\\` 就含中文）。改用 np.fromfile + cv2.imdecode 规避。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_bgr(path: str | Path, img: np.ndarray) -> bool:
    """写出图片（同样规避中文路径）：cv2.imencode + ndarray.tofile。"""
    suffix = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _resize_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def read_image(path: str, max_side: int = DEFAULT_MAX_SIDE) -> np.ndarray:
    """读取单张图片为 BGR，长边缩放到 max_side（等比）。"""
    img = imread_bgr(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return _resize_long_side(img, max_side)


class SingleFrameReader:
    """单帧读取器（图片）：把 ndarray 包装成 FrameReader，与视频共用同一套循环。

    图片本质上就是「只有一帧的流」，单独写分支会让上层出现
    `if isinstance(src, np.ndarray)` 这类分叉；包装后上层只认协议。
    """

    fps: float | None = None  # 图片无帧率概念 → 由调用方退回挂钟计时

    def __init__(self, img: np.ndarray):
        self._img: np.ndarray | None = img
        self._consumed = False

    def read(self) -> np.ndarray:
        if self._consumed or self._img is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        self._consumed = True
        return self._img

    def close(self) -> None:
        self._img = None


class VideoReader:
    """视频文件 / 摄像头 / RTSP 流读取器（cv2.VideoCapture 统一封装）。"""

    def __init__(self, source: str | int, max_side: int = DEFAULT_MAX_SIDE):
        # 纯数字视为本机摄像头索引（cv2 传字符串 "0" 会被当成文件名）
        self.source = int(source) if isinstance(source, int) or str(source).isdigit() else str(source)
        self.max_side = max_side
        self.backend = ""
        self._cap: cv2.VideoCapture | None = None
        if isinstance(self.source, int):
            self._open_camera(self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

    def _open_camera(self, index: int) -> None:
        """按候选后端依次打开摄像头，并记录实际生效的后端名（供日志/接口排查）。"""
        failed: list[str] = []
        for backend in _camera_backend_list():
            cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if cap.isOpened():
                self._cap = cap
                self.backend = _backend_label(backend)
                log.info("[Reader] 摄像头 %d 打开成功（后端 %s）", index, self.backend)
                return
            cap.release()
            failed.append(_backend_label(backend))
        log.warning("[Reader] 摄像头 %d 在所有候选后端上都打不开: %s", index, failed)

    @property
    def fps(self) -> float | None:
        """视频帧率；拿不到或异常值返回 None（调用方退回挂钟计时）。"""
        try:
            val = float(self._cap.get(cv2.CAP_PROP_FPS))
        except Exception:  # 某些流不返回该属性
            return None
        return val if val > 0 else None

    @property
    def frame_count(self) -> int:
        """容器声称的总帧数；拿不到或异常值返回 0（调用方据此判断能否报进度百分比）。

        注意这只是**元数据**：实测合成视频回读时它准，但被截断/损坏的文件会偏大，
        所以进度条只能当参考，最终以实际处理帧数为准。
        """
        try:
            val = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:  # 某些流（RTSP 直播）不返回该属性
            return 0
        return val if val > 0 else 0

    def read(self) -> np.ndarray:
        """读取下一帧；结束返回空数组。"""
        ok, frame = self._cap.read()
        if not ok:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return _resize_long_side(frame, self.max_side)

    def skip(self, n: int = 1) -> int:
        """丢弃 n 帧（只解码不取图），返回实际丢弃数。

        直播源（摄像头 / RTSP）落后时用它追赶：宁可不看中间几帧，也不要越播延迟越大。
        对文件源请慎用——那等于跳过内容。
        """
        dropped = 0
        for _ in range(max(0, int(n))):
            if not self._cap.grab():
                break
            dropped += 1
        return dropped

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame.size == 0:
            raise StopIteration
        return frame

    def close(self) -> None:
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
            self._cap = None


def probe_cameras(indices: int | list[int] = 4) -> list[dict]:
    """探测本机摄像头：**能打开且能读到一帧**才算可用。

    "能打开但读不到帧"的设备（虚拟摄像头 / 被其它程序占用 / 驱动未就绪）最容易骗人——
    页面上就是选了个"看起来存在"的摄像头然后一直黑屏。因此这里必须真的读一帧才算数。

    返回每项的 `{index, ok, backend, first_frame_ms, width, height, error}`。

    ⚠️ 单个设备的 `read()` 可能长时间阻塞（驱动问题），**调用方必须自己加超时**
    （服务端是把本函数丢进线程池 + `wait_for` 限时）。
    """
    if isinstance(indices, int):
        indices = list(range(indices))
    results: list[dict] = []
    for index in indices:
        rec = {"index": index, "ok": False, "backend": "", "first_frame_ms": 0.0,
               "width": 0, "height": 0, "error": ""}
        cap = None
        try:
            for backend in _camera_backend_list():
                cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    rec["backend"] = _backend_label(backend)
                    break
                cap.release()
                cap = None
            if cap is None:
                rec["error"] = "所有候选后端都打不开"
                results.append(rec)
                continue

            t0 = time.perf_counter()
            ok, frame = cap.read()
            rec["first_frame_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if ok and frame is not None and getattr(frame, "size", 0):
                rec["ok"] = True
                rec["height"], rec["width"] = frame.shape[:2]
            else:
                rec["error"] = "能打开但读不到帧（设备被占用 / 无输出）"
        except Exception as exc:  # noqa: BLE001 —— 驱动层的异常类型不可预期，一律记进结果
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if cap is not None:
                cap.release()
        results.append(rec)
    return results


def make_reader(source: str | int, max_side: int = DEFAULT_MAX_SIDE) -> SingleFrameReader | VideoReader:
    """按来源返回 FrameReader：图片 → SingleFrameReader，视频/摄像头/RTSP → VideoReader。"""
    text = str(source)
    if not text.isdigit():
        path = Path(text)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return SingleFrameReader(read_image(text, max_side))
        if not path.is_file():
            log.debug("[Reader] %s 不是本地文件，按视频流处理", text)
    return VideoReader(source, max_side)


def iter_frames(reader) -> Iterator[np.ndarray]:
    """逐帧产出 BGR 帧，结束时（含异常、提前 break）自动释放资源。"""
    try:
        while True:
            frame = reader.read()
            if frame is None or getattr(frame, "size", 0) == 0:
                break
            yield frame
    finally:
        reader.close()
