"""车辆检测评测（mAP）。计划书 §6 M2 / §8 指标。

用法：python tools/eval_vehicle.py --weights weights/vehicle_best.pt --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import CONFIGS_DIR  # noqa: E402
from src.common.logger import setup_logger  # noqa: E402


def main() -> int:
    setup_logger("eval_vehicle")
    parser = argparse.ArgumentParser(description="评测车辆检测模型")
    parser.add_argument("--weights", default="weights/vehicle_best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(CONFIGS_DIR / "vehicle_yolov8s.yaml"),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
    )

    logging.info("[Eval] split=%s mAP@0.5=%.4f mAP@0.5:0.95=%.4f",
                 args.split, float(metrics.box.map50), float(metrics.box.map))
    return 0


if __name__ == "__main__":
    sys.exit(main())
