"""LPRNet 字符识别网络 + CTC 解码。

计划书 §6 M3：输入 94×24，字符集 68 类（31 省份 + 字母 + 数字 + blank），
CTC Loss + greedy decode。此实现为「简化版 LPRNet」，核心思想一致：
CNN 下采样宽度方向保留时间步 T，输出 (B, T, C) 供 CTC 对齐。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# 字符集：31 省份简称 + 24 字母（去 I/O）+ 10 数字
# 注意：省份简称必须与 src/postprocess/rules.py 的 PROVINCES 保持一致
PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
CHARS = PROVINCES + "ABCDEFGHJKLMNPQRSTUVWXYZ" + "0123456789"

# blank 索引约定为 0，字符表不含 blank（CTC 输出类别数 = len(CHARS) + 1）
BLANK_IDX = 0
NUM_CLASSES = len(CHARS) + 1  # 含 blank

# 输出时间步 T：宽度 94 经两次 stride=2 下采样 → 94/4 = 23（下取整后对齐）
SEQ_LEN = 23


class SmallBasicBlock(nn.Module):
    """LPRNet 的 basic block：1×1 降通道 + 3×3 卷积 + 残差。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch // 4, 1)
        self.bn1 = nn.BatchNorm2d(out_ch // 4)
        self.conv2 = nn.Conv2d(out_ch // 4, out_ch // 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch // 4)
        self.conv3 = nn.Conv2d(out_ch // 4, out_ch, 1)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.shortcut(x))


class LPRNet(nn.Module):
    """简化版 LPRNet：输入 (B, 3, 24, 94)，输出 (B, SEQ_LEN, NUM_CLASSES)。

    高度方向 24 → 12 → 6 后全局平均池化；宽度方向 94 → 47 → 23 保留时间步。
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # 47×12
            SmallBasicBlock(64, 128),
            nn.MaxPool2d((2, 1)),  # 23×12（仅压宽度）
            SmallBasicBlock(128, 128),
            nn.MaxPool2d((1, 2)),  # 23×6（仅压高度）
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.stem(x)            # (B, 128, 6, 23)
        out = self.head(f)          # (B, C, 6, 23)
        out = out.mean(dim=2)       # 高度方向全局平均池化 → (B, C, 23)
        out = out.permute(0, 2, 1)  # (B, T=23, C)
        return out


def ctc_greedy_decode(logits: np.ndarray, blank: int = BLANK_IDX) -> str:
    """CTC greedy decode：逐时间步取 argmax，合并相邻重复，剔除 blank。

    Args:
        logits: (T, C) 的 log-prob 或 logits 矩阵。
        blank: blank 类别索引。

    Returns:
        解码出的字符串（不含 blank）。
    """
    idxs = np.argmax(logits, axis=1)
    decoded: list[int] = []
    prev = blank
    for idx in idxs:
        if idx != blank and idx != prev:
            decoded.append(int(idx))
        prev = idx
    return "".join(CHARS[i - 1] for i in decoded if 1 <= i < len(CHARS) + 1)


def _softmax(x: np.ndarray) -> np.ndarray:
    """按最后一维做数值稳定 softmax。"""
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def ctc_greedy_decode_with_score(
    logits: np.ndarray, blank: int = BLANK_IDX
) -> tuple[str, float]:
    """CTC greedy decode，同时给出置信度。

    置信度 = 被保留字符在其时间步上的 softmax 概率均值；无字符输出时为 0.0。
    随机初始化权重下该值接近 1/NUM_CLASSES（≈0.015），训练后正确样本应显著更高。

    Returns:
        (车牌字符串, 置信度 0~1)
    """
    probs = _softmax(logits)
    idxs = np.argmax(logits, axis=1)
    chars: list[str] = []
    confs: list[float] = []
    prev = blank
    for t, idx in enumerate(idxs):
        if idx != blank and idx != prev and 1 <= idx < len(CHARS) + 1:
            chars.append(CHARS[idx - 1])
            confs.append(float(probs[t, idx]))
        prev = idx
    if not confs:
        return "", 0.0
    return "".join(chars), sum(confs) / len(confs)


def decode_batch(logits: np.ndarray, blank: int = BLANK_IDX) -> list[str]:
    """批量解码 (B, T, C)。"""
    return [ctc_greedy_decode(logits[i], blank) for i in range(logits.shape[0])]
