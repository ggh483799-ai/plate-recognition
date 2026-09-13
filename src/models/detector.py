"""YOLO 检测器封装（Ultralytics）。计划书 §3.2：无状态、可并发。"""

from __future__ import annotations

import logging

import numpy as np

from src.common.interfaces import BBox

log = logging.getLogger(__name__)


class YoloDetector:
    """封装 Ultralytics YOLO 推理，输出 BBox 列表。"""

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        # 延迟 import：ultralytics 安装较慢，允许模块在未装时被 import
        from ultralytics import YOLO

        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(weights)
        # 模型自带类别表。预训练 COCO 与自训练模型的类别序**不同**（COCO 里 cls=2 是 car，
        # 自训练 4 类里 cls=2 是 truck），因此车辆类型必须按这份表解析，不能硬编码映射。
        self.names = dict(getattr(self.model, "names", {}) or {})
        log.info("[Detector] 加载模型: %s (device=%s, 类别数=%d)", weights, device, len(self.names))

    def detect(self, frame: np.ndarray) -> list[BBox]:
        """对单帧推理，返回按置信度降序的 BBox 列表。"""
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        boxes: list[BBox] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, clss):
                boxes.append(BBox(float(x1), float(y1), float(x2), float(y2), float(score), int(cls)))
        boxes.sort(key=lambda b: b.score, reverse=True)
        return boxes
