"""把一张车牌图合成一段「运动」视频，用于在无摄像头 / 无真实素材时验证视频链路。

用途
----
1. 验证 `src/pipeline/video_pipeline.py`（跨帧去重、落盘、标注视频）；
2. 没有摄像头时给 MV-03（多路监控）当模拟源：`mediamtx` 把生成的 mp4 推成 RTSP 即可。

做法：把图片等比放大到比输出画面略大，再让取样窗口**匀速平移 + 轻微缩放**，
于是车牌在画面中的坐标逐帧变化，等价于车辆在动——不是简单复制同一帧 N 次
（那样无法验证「位置变化但车牌号不变仍应去重成一个事件」）。

    python tools/make_demo_video.py --img data/field/test_car.png --out runs/video_e2e/demo.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logger import setup_logger  # noqa: E402
from src.io.reader import imread_bgr  # noqa: E402
from src.io.writer import AnnotatedVideoWriter  # noqa: E402

log = logging.getLogger(__name__)


def make_video(img: np.ndarray, out: str, seconds: float, fps: float,
               size: tuple[int, int], zoom: float) -> int:
    """把单图合成为运动视频，返回写出帧数。

    **必须等比放置（letterbox），不能拉伸**
    --------------------------------------
    初版用 `cv2.resize(img, (w*zoom, h*zoom))` 直接把图拉成画面比例。源图是 350×562 的竖构图，
    被横向拉伸成 1472×828 后车牌严重变形——实测该帧送入引擎直接**读不出车牌**（原图 0.8567/0.9987，
    拉伸后为空）。所以这里改为「等比缩放装进放大画布 + 灰底补齐」，再在放大画布内平移取样。

    参数
    ----
    size : 输出画面 (宽, 高)
    zoom : 放大倍数（>1 才有平移余量）
    """
    w, h = size
    bw, bh = int(w * zoom), int(h * zoom)      # 放大画布（平移余量来自这里）
    iw, ih = img.shape[1], img.shape[0]
    scale = min(bw / iw, bh / ih)              # 等比装进放大画布
    rw, rh = max(1, int(iw * scale)), max(1, int(ih * scale))
    base = np.full((bh, bw, 3), 32, dtype=np.uint8)   # 灰底，避免纯黑影响画面对比度
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    ox, oy = (bw - rw) // 2, (bh - rh) // 2
    base[oy:oy + rh, ox:ox + rw] = resized

    total = max(1, int(round(seconds * fps)))
    writer = AnnotatedVideoWriter(out, fps=fps)

    for i in range(total):
        ratio = i / max(1, total - 1)
        x = int((bw - w) * ratio)              # 水平匀速平移（模拟车辆横向移动）
        y = int((bh - h) * 0.5)
        frame = base[y:y + h, x:x + w]
        if frame.shape[:2] != (h, w):          # 兜底：绝不把错尺寸的帧交给 VideoWriter
            log.warning("[DemoVideo] 帧尺寸异常 %s，已纠正为 %s", frame.shape[:2], (h, w))
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame.copy())

    writer.close()
    return writer.frames_written

    writer.close()
    return writer.frames_written


def main() -> int:
    setup_logger("make_demo_video")
    parser = argparse.ArgumentParser(description="由单张图合成运动视频（视频链路验证用）")
    parser.add_argument("--img", required=True, help="源图路径")
    parser.add_argument("--out", required=True, help="输出 mp4 路径")
    parser.add_argument("--seconds", type=float, default=3.0, help="时长（秒）")
    parser.add_argument("--fps", type=float, default=10.0, help="帧率")
    parser.add_argument("--width", type=int, default=1280, help="输出画面宽")
    parser.add_argument("--height", type=int, default=720, help="输出画面高")
    parser.add_argument("--zoom", type=float, default=1.15, help="预放大倍数（平移余量）")
    args = parser.parse_args()

    img = imread_bgr(args.img)
    if img is None:
        logging.error("[DemoVideo] 无法读取源图: %s", args.img)
        return 1
    if args.zoom <= 1.0:
        logging.error("[DemoVideo] --zoom 必须 > 1，否则没有平移余量")
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = make_video(img, args.out, args.seconds, args.fps, (args.width, args.height), args.zoom)
    logging.info("[DemoVideo] 已写出 %d 帧 → %s", n, args.out)
    if n == 0:
        logging.error("[DemoVideo] 未写出任何帧")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
