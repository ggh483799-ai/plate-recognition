"""LPRNet 推理封装（实现 Recognizer 协议）。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


class LprNetRecognizer:
    """封装 LPRNet 推理：预处理 → 前向 → CTC greedy decode。

    权重缺失时使用随机初始化（仅验证链路，不追求精度），
    训练权重存在时自动加载。
    """

    def __init__(self, weights: str | None = None, device: str = "cpu"):
        from src.models.lprnet import LPRNet, ctc_greedy_decode_with_score

        self._decode_with_score = ctc_greedy_decode_with_score
        self.device = device
        self.net = LPRNet()
        loaded = False
        if weights and Path(weights).exists():
            state = torch.load(weights, map_location="cpu")
            self.net.load_state_dict(state)
            loaded = True
            log.info("[Recognizer] 加载 LPRNet 权重: %s", weights)
        else:
            log.warning("[Recognizer] 未找到权重 %s，使用随机初始化（仅链路验证）", weights)
        self.net.eval()
        self.loaded = loaded

    def recognize(self, plate_img: np.ndarray) -> str:
        """识别单张车牌小图，返回字符串（Recognizer 协议实现）。"""
        return self.recognize_with_confidence(plate_img)[0]

    def recognize_with_confidence(self, plate_img: np.ndarray) -> tuple[str, float]:
        """识别单张车牌小图，返回 (车牌字符串, 置信度)。

        置信度为 CTC 保留字符上的 softmax 概率均值，可直接暴露给上层展示，
        替代早期版本硬编码阈值的做法。
        """
        img = cv2.resize(plate_img, (94, 24))
        tensor = (
            torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        )
        with torch.no_grad():
            logits = self.net(tensor)[0].numpy()  # (T, C)
        return self._decode_with_score(logits)
