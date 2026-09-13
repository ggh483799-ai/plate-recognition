"""跨模块接口定义（typing.Protocol + dataclass），便于替换实现。

计划书 §9.1：跨模块接口统一在此定义，各实现类只需满足 Protocol 即可替换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BBox:
    """检测框（像素坐标 xyxy），附带类别与置信度。"""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    cls: int = 0

    @property
    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass
class PlateResult:
    """单个车牌的识别结果（流水线输出单元）。"""

    plate_no: str = ""
    plate_color: str = ""
    vehicle_type: str = ""
    det_score: float = 0.0
    rec_score: float = 0.0
    bbox: list[float] = field(default_factory=list)
    cost_ms: float = 0.0

    def to_dict(self) -> dict:
        # 统一 float() 强转：模型输出常为 numpy 标量，round(np.float32) 仍是 np.float32，
        # 直接 json.dumps 会报 "Object of type float32 is not JSON serializable"
        return {
            "plate_no": self.plate_no,
            "plate_color": self.plate_color,
            "vehicle_type": self.vehicle_type,
            "det_score": round(float(self.det_score), 4),
            "rec_score": round(float(self.rec_score), 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "cost_ms": round(float(self.cost_ms), 2),
        }


# ============================================================
# 接口协议
# ============================================================

class Detector(Protocol):
    """目标检测器：输入 BGR 帧，输出检测框列表。无状态、可并发。"""

    def detect(self, frame: np.ndarray) -> list[BBox]:
        """对单帧做推理，返回按置信度降序的框列表。"""
        ...


class Recognizer(Protocol):
    """字符识别器：输入车牌小图（BGR），输出车牌字符串。"""

    def recognize(self, plate_img: np.ndarray) -> str:
        """识别单张车牌图，返回车牌号字符串。"""
        ...


class FrameReader(Protocol):
    """帧读取器：统一读取图片 / 视频 / RTSP 流，输出 BGR 帧。"""

    def read(self) -> np.ndarray:
        """读取下一帧；无更多帧时返回空数组。"""
        ...

    def close(self) -> None:
        """释放资源。"""
        ...
