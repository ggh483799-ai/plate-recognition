"""LPRNet 字符识别训练（自实现 CTC）。计划书 §6 M3。

数据：车牌字符图，文件名即标签（京A12345_*.jpg），见计划书 §4.3 文件名约定。
损失：CTC Loss；解码：greedy decode。
⚠️ 需 GPU 环境执行；无 GPU 用 --device cpu --epochs 1 仅验证脚本可跑通。

用法：python tools/train_lprnet.py --img-dir data/raw/crpd --device 0 --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.common.config import WEIGHTS_DIR  # noqa: E402
from src.common.logger import setup_logger  # noqa: E402
from src.io.reader import imread_bgr  # noqa: E402
from src.models.lprnet import CHARS, LPRNet, SEQ_LEN, ctc_greedy_decode  # noqa: E402
from src.postprocess.rules import extract_plate_candidate  # noqa: E402


def encode_label(plate: str) -> list[int]:
    """车牌号 → 字符索引序列（+1 偏移，0 留给 blank）。"""
    idxs = []
    for c in plate:
        if c in CHARS:
            idxs.append(CHARS.index(c) + 1)
    return idxs


class PlateDataset(Dataset):
    """车牌图片数据集：从文件名提取标签。"""

    def __init__(self, img_dir: str):
        self.img_dir = Path(img_dir)
        self.files = sorted(
            p for p in self.img_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = imread_bgr(path)
        if img is None:
            raise FileNotFoundError(f"无法读取车牌图: {path}")
        img = cv2.resize(img, (94, 24))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        label = encode_label(extract_plate_candidate(path.stem))
        return tensor, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    target = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return imgs, target, target_lengths


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, cnt = 0.0, 0
    for imgs, target, target_lengths in loader:
        imgs = imgs.to(device)
        target = target.to(device)
        target_lengths = target_lengths.to(device)
        batch = imgs.size(0)
        logits = model(imgs)                                   # (B, T, C)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)      # (T, B, C)
        input_lengths = torch.full((batch,), SEQ_LEN, dtype=torch.long, device=device)
        loss = criterion(log_probs, target, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * batch
        cnt += batch
    return total / max(cnt, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """返回 (单字符准确率, 整牌准确率)。"""
    model.eval()
    char_ok = char_total = plate_ok = plate_total = 0
    for imgs, target, target_lengths in loader:
        imgs = imgs.to(device)
        logits = model(imgs)                        # (B, T, C)
        for i in range(imgs.size(0)):
            pred = ctc_greedy_decode(logits[i].cpu().numpy())
            gt = "".join(
                CHARS[t - 1] for t in target[sum(target_lengths[:i]): sum(target_lengths[:i + 1])]
            )
            char_total += len(gt)
            # 字符级：按编辑距离近似（简化：逐字符对比到较短长度）
            char_ok += sum(1 for a, b in zip(pred, gt) if a == b)
            plate_total += 1
            if pred == gt:
                plate_ok += 1
    return (char_ok / char_total if char_total else 0.0,
            plate_ok / plate_total if plate_total else 0.0)


def main() -> int:
    setup_logger("train_lprnet")
    parser = argparse.ArgumentParser(description="训练 LPRNet 字符识别模型")
    parser.add_argument("--img-dir", required=True, help="车牌字符图目录（文件名即标签）")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    logging.info("[Train] device=%s epochs=%d batch=%d", device, args.epochs, args.batch)

    dataset = PlateDataset(args.img_dir)
    if len(dataset) == 0:
        logging.error("[Train] 数据目录为空: %s", args.img_dir)
        return 1
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    model = LPRNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        char_acc, plate_acc = evaluate(model, loader, device)
        logging.info(
            "[Train] epoch=%d loss=%.4f char_acc=%.4f plate_acc=%.4f",
            epoch, loss, char_acc, plate_acc,
        )

    torch.save(model.state_dict(), WEIGHTS_DIR / "lprnet_best.pt")
    logging.info("[Train] 权重已保存: %s", WEIGHTS_DIR / "lprnet_best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
