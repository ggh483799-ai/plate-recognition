"""车牌底色判定（HSV 阈值法）。纯函数，可脱离模型单测。

计划书 §4.3：车牌颜色 blue/green/yellow/white/black 五类。
"""

from __future__ import annotations

import cv2
import numpy as np


def plate_color(img_bgr: np.ndarray) -> str:
    """按 HSV 阈值判定车牌底色。

    取车牌中央区域（避开边缘与字符）统计主色调。
    返回：blue / green / yellow / white / black。
    """
    h, w = img_bgr.shape[:2]
    # 取中央 60% 区域，避开边框与字符干扰
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    cy0, cy1 = int(h * 0.2), int(h * 0.8)
    if cx1 <= cx0 or cy1 <= cy0:
        return "unknown"
    roi = img_bgr[cy0:cy1, cx0:cx1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = hh.size

    # 各类颜色像素占比（OpenCV HSV：H∈[0,180), S/V∈[0,255]）
    blue = np.sum((hh >= 100) & (hh <= 124) & (ss >= 43) & (vv >= 46))
    green = np.sum((hh >= 35) & (hh <= 77) & (ss >= 43) & (vv >= 46))
    yellow = np.sum((hh >= 11) & (hh <= 34) & (ss >= 43) & (vv >= 46))
    white = np.sum((ss < 43) & (vv > 150))
    black = np.sum((vv < 60))

    ratios = {
        "blue": blue / total,
        "green": green / total,
        "yellow": yellow / total,
        "white": white / total,
        "black": black / total,
    }
    best = max(ratios, key=ratios.get)
    # 主色占比过低则视为未知
    return best if ratios[best] >= 0.30 else "unknown"
