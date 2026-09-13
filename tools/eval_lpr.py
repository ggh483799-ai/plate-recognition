"""端到端车牌识别评测。计划书 §6 M3 / §8：整牌准确率 + 单字符准确率。

输入目录内图片文件名即车牌号（京A12345_*.jpg），逐张跑 pipeline 后比对。

用法：python tools/eval_lpr.py --img-dir data/field --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.common.logger import setup_logger  # noqa: E402
from src.io.reader import imread_bgr  # noqa: E402
from src.postprocess.rules import extract_plate_candidate  # noqa: E402


def char_accuracy(pred: str, gt: str) -> float:
    """字符级准确率（1 - 编辑距离/长度，简化为逐位比对）。"""
    if not gt:
        return 0.0
    correct = sum(1 for a, b in zip(pred, gt) if a == b)
    return correct / len(gt)


def main() -> int:
    setup_logger("eval_lpr")
    parser = argparse.ArgumentParser(description="端到端车牌识别评测")
    parser.add_argument("--img-dir", required=True, help="测试图目录（文件名即车牌号）")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from src.models.detector import YoloDetector
    from src.models.lpr_recognizer import LprNetRecognizer
    from src.pipeline.lpr_pipeline import LprPipeline

    img_dir = Path(args.img_dir)
    files = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not files:
        logging.error("[Eval] 目录为空: %s", args.img_dir)
        return 1

    vehicle = YoloDetector("yolov8n.pt", device=args.device)
    plate = YoloDetector("yolov8n.pt", device=args.device)
    recognizer = LprNetRecognizer(None, device=args.device)
    pipeline = LprPipeline(vehicle, plate, recognizer)

    plate_total = plate_ok = 0
    char_total = char_ok = 0
    for p in files:
        gt = extract_plate_candidate(p.stem)
        img = imread_bgr(p)
        if img is None:
            continue
        results = pipeline.run(img)
        pred = results[0].plate_no if results else ""
        plate_total += 1
        if pred == gt:
            plate_ok += 1
        char_total += len(gt)
        char_ok += sum(1 for a, b in zip(pred, gt) if a == b)

    plate_acc = plate_ok / plate_total if plate_total else 0.0
    char_acc = char_ok / char_total if char_total else 0.0
    logging.info("[Eval] 整牌准确率=%.4f (%d/%d)", plate_acc, plate_ok, plate_total)
    logging.info("[Eval] 单字符准确率=%.4f (%d/%d)", char_acc, char_ok, char_total)
    print(f"plate_acc={plate_acc:.4f} char_acc={char_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
