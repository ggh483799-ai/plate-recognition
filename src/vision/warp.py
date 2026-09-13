"""车牌四点透视矫正，统一尺寸 94×24。纯函数，可脱离模型单测。"""

from __future__ import annotations

import cv2
import numpy as np

# 国内蓝牌标准比例（计划书 §6 M3：输入尺寸 94×24）
PLATE_W = 94
PLATE_H = 24


def order_points(pts: np.ndarray) -> np.ndarray:
    """四点排序为 [左上, 右上, 右下, 左下]。

    依据：左上角 x+y 最小，右下角 x+y 最大，右上角 y-x 最小。
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    rect[1] = pts[np.argmin(d)]  # 右上
    rect[3] = pts[np.argmax(d)]  # 左下
    return rect


def warp_plate(
    img: np.ndarray,
    pts: np.ndarray,
    size: tuple[int, int] = (PLATE_W, PLATE_H),
) -> np.ndarray:
    """把四点框出的车牌区域透视矫正到统一尺寸 (w, h)。

    Args:
        img: BGR 图像。
        pts: 4 个角点（任意顺序，内部自动排序）。
        size: 目标 (宽, 高)，默认 94×24。

    Returns:
        矫正后的车牌小图（BGR）。
    """
    w, h = int(size[0]), int(size[1])
    src = order_points(pts)
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h))


def pad_bbox(bbox: tuple[float, float, float, float], ratio: float = 0.1) -> np.ndarray:
    """外扩检测框（计划书 §3.1：逐车裁剪外扩 10% padding），返回 4 点。

    bbox 为 (x1, y1, x2, y2)，ratio 为每边外扩比例。
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    px, py = w * ratio, h * ratio
    # 返回 [左上, 右上, 右下, 左下] 四角点
    return np.array(
        [
            [x1 - px, y1 - py],
            [x2 + px, y1 - py],
            [x2 + px, y2 + py],
            [x1 - px, y2 + py],
        ],
        dtype=np.float32,
    )
