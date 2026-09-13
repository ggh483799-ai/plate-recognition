"""识别结果可视化：按车牌底色分色画框 + 车牌号标签。

为什么需要 Pillow
-----------------
`cv2.putText` 只支持 ASCII（Hershey 点阵字体），写中文会渲染成 `????`。
而车牌号**必然含中文**（省份简称，如「粤」），标签又是给人看的核心信息。
因此标签走 Pillow + 系统中文字体；Pillow 或字体任一缺失时**降级为 ASCII 标签**
（框、置信度仍在，只是丢掉中文），不抛异常、不阻断视频输出。

本模块只做绘制，不做识别；函数为纯函数（`inplace=False` 时不改写入参）。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ===== 车牌底色 → BGR 画框色（键与 PlateResult.plate_color 取值一致）=====
PLATE_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "blue": (255, 128, 0),      # 蓝牌（小型车）
    "green": (0, 200, 0),       # 绿牌（新能源）
    "yellow": (0, 200, 255),    # 黄牌（大车/教练/挂车）
    "white": (235, 235, 235),   # 白牌（警/军）
    "black": (90, 90, 90),      # 黑牌（港澳/使领馆）——纯黑在暗底上看不见，提亮成深灰
}
DEFAULT_COLOR_BGR = (0, 220, 255)  # unknown / 空值兜底（橙黄，与上面几色都不撞）

# 底色中文名（用于标签展示）
COLOR_LABEL_CN: dict[str, str] = {
    "blue": "蓝牌",
    "green": "绿牌",
    "yellow": "黄牌",
    "white": "白牌",
    "black": "黑牌",
}

# 候选中文字体（Windows 自带，按优先级）
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/simsun.ttc",  # 宋体
)


def color_for(plate_color: str) -> tuple[int, int, int]:
    """车牌底色 → 画框 BGR 颜色。"""
    return PLATE_COLOR_BGR.get(plate_color, DEFAULT_COLOR_BGR)


@lru_cache(maxsize=8)
def _load_font(px: int):
    """加载中文字体；Pillow 或字体不可用时返回 None（调用方降级为 ASCII）。"""
    try:
        from PIL import ImageFont
    except ImportError:
        log.warning("[Draw] Pillow 不可用，标签降级为 ASCII（cv2.putText）")
        return None
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, px)
            except OSError as exc:  # 字体文件损坏/不被 FreeType 支持
                log.debug("[Draw] 字体加载失败 %s: %s", path, exc)
    log.warning("[Draw] 未找到候选中文字体 %s，标签降级为 ASCII", _FONT_CANDIDATES)
    return None


def _ascii_only(text: str) -> str:
    """只保留 ASCII 可打印字符（cv2.putText 的降级路径用）。"""
    return "".join(ch for ch in text if 32 <= ord(ch) < 127) or "plate"


# 对外别名：实时页的物体框标签要用同一套字体逻辑（避免跨模块引用私有名）
load_font = _load_font


def _text_color_for(bg_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """按底色亮度选文字颜色（深底配白字，浅底配黑字），保证可读。"""
    b, g, r = bg_bgr
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    return (255, 255, 255) if luma < 140 else (0, 0, 0)


def _text_tile(text: str, font, bg_bgr: tuple[int, int, int]) -> np.ndarray:
    """用 Pillow 把一行中文标签渲染成 BGR 小图块（避免每帧整图 RGB 互转）。

    只渲染标签大小的小图，再靠 numpy 切片贴回原帧——比整帧 `cv2→PIL→cv2` 快两个数量级。
    """
    from PIL import Image, ImageDraw

    rgb_bg = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
    rgb_fg = _text_color_for(bg_bgr)[::-1]

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = probe.textbbox((0, 0), text, font=font)
    pad_x, pad_y = 6, 4
    w = max(1, box[2] - box[0]) + pad_x * 2
    h = max(1, box[3] - box[1]) + pad_y * 2

    tile = Image.new("RGB", (w, h), rgb_bg)
    ImageDraw.Draw(tile).text((pad_x - box[0], pad_y - box[1]), text, font=font, fill=rgb_fg)
    return cv2.cvtColor(np.asarray(tile), cv2.COLOR_RGB2BGR)


def _label_text(res) -> str:
    """标签文案：车牌号 · 底色 · 识别置信度。"""
    plate_no = getattr(res, "plate_no", "") or "—"
    plate_color = getattr(res, "plate_color", "") or ""
    cn = COLOR_LABEL_CN.get(plate_color, plate_color or "未知")
    return f"{plate_no} · {cn} · {float(getattr(res, 'rec_score', 0.0)):.0%}"


def _draw_one(canvas: np.ndarray, res, font) -> None:
    """在画布上绘制单个车牌：矩形框 + 标签。越界/非法框直接跳过。"""
    h, w = canvas.shape[:2]
    box = list(getattr(res, "bbox", None) or [])
    if len(box) != 4:
        return
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        log.debug("[Draw] 非法车牌框，跳过绘制: %s", box)
        return

    color = color_for(getattr(res, "plate_color", ""))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    text = _label_text(res)
    if font is None:  # 降级：ASCII 标签
        cv2.putText(canvas, _ascii_only(text), (x1, max(14, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        return

    tile = _text_tile(text, font, color)
    th, tw = tile.shape[:2]
    ty = y1 - th if y1 - th >= 0 else y2  # 框上方放不下就贴框内下沿
    tx = max(0, min(x1, w - tw))
    th_eff, tw_eff = min(th, h - ty), min(tw, w - tx)
    if th_eff <= 0 or tw_eff <= 0:
        return
    canvas[ty:ty + th_eff, tx:tx + tw_eff] = tile[:th_eff, :tw_eff]


def crop_plate(frame: np.ndarray, bbox, margin: float = 0.18) -> np.ndarray | None:
    """按检测框裁出车牌小图（外扩 margin 保留一点上下文）。非法框返回 None。

    **必须 copy()**：`cv2.VideoCapture.read()` 复用同一块帧缓冲，直接切片拿到的是视图，
    下一帧读入后视图内容会被覆盖——存下来的"证据图"会变成别的画面。
    """
    box = list(bbox or [])
    if len(box) != 4 or frame is None or getattr(frame, "size", 0) == 0:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    xi1, yi1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    xi2, yi2 = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    if xi2 - xi1 < 2 or yi2 - yi1 < 2:
        return None
    return frame[yi1:yi2, xi1:xi2].copy()


def draw_hud(canvas: np.ndarray, lines: list[str], font=None) -> None:
    """左上角 HUD 信息（帧号 / 速度 / 命中数等）。原地写入。"""
    if not lines:
        return
    text = "  ".join(lines)
    if font is None:
        y = 22
        for line in lines:
            cv2.putText(canvas, _ascii_only(line), (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y += 24
        return
    tile = _text_tile(text, font, (40, 40, 40))
    th, tw = tile.shape[:2]
    h, w = canvas.shape[:2]
    canvas[:min(th, h), :min(tw, w)] = tile[:min(th, h), :min(tw, w)]


# 物体框颜色（BGR）：车辆/行人用细框 + 类别名，与车牌的"粗实线 + 号牌"区分开
OBJECT_COLOR_BY_NAME: dict[str, tuple[int, int, int]] = {
    "person": (0, 170, 255),        # 橙
    "car": (110, 200, 60),          # 草绿
    "bus": (200, 130, 240),
    "truck": (150, 150, 60),
    "motorcycle": (240, 130, 200),
}
OBJECT_COLOR_DEFAULT = (200, 200, 200)
# 灰色：检测到了但没过阈值的"疑似车牌"（玩具车/小车牌最常见）
CANDIDATE_COLOR = (170, 170, 170)


def object_color(name: str) -> tuple[int, int, int]:
    return OBJECT_COLOR_BY_NAME.get(name, OBJECT_COLOR_DEFAULT)


def draw_objects(canvas: np.ndarray, objects, font=None, inplace: bool = True) -> np.ndarray:
    """画"物体框"：细线 + 文字标签。`objects = [(bbox, 文本, BGR颜色)]`。

    与车牌框（`draw_detections`，粗实线 + 号牌）刻意区分开，一眼能分清"这是框人/框车"
    还是"这是识别到的车牌"。文本走 Pillow（与车牌标签同一套字体逻辑），缺失时降级 ASCII。
    """
    items = list(objects or [])
    out = canvas if inplace else canvas.copy()
    h, w = out.shape[:2]
    for obj in items:
        if not obj or len(obj) < 3:
            continue
        box, text, color = obj[0], obj[1], obj[2]
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        if not text:
            continue
        if font is None:
            cv2.putText(out, _ascii_only(text), (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            continue
        tile = _text_tile(text, font, color)
        th, tw = tile.shape[:2]
        ty = y1 - th if y1 - th >= 0 else y2
        tx = max(0, min(x1, w - tw))
        th_eff, tw_eff = min(th, h - ty), min(tw, w - tx)
        if th_eff <= 0 or tw_eff <= 0:
            continue
        out[ty:ty + th_eff, tx:tx + tw_eff] = tile[:th_eff, :tw_eff]
    return out


def draw_detections(
    frame: np.ndarray,
    results,
    inplace: bool = False,
    hud: list[str] | None = None,
) -> np.ndarray:
    """把识别结果画到帧上，返回标注图。

    参数
    ----
    frame   : BGR 帧
    results : PlateResult 序列（可为空）
    inplace : True 直接改写入参（省一次拷贝，用于只做标注视频的场景）
    hud     : 左上角附加信息行（可选）
    """
    results = list(results or [])
    canvas = frame if inplace else frame.copy()
    if results:
        # 字号随画面宽度自适应，避免 4K 图上的标签小到看不见
        px = int(np.clip(canvas.shape[1] / 55, 14, 40))
        font = _load_font(px)
        for res in results:
            _draw_one(canvas, res, font)
    if hud:
        draw_hud(canvas, hud, _load_font(20))
    return canvas
