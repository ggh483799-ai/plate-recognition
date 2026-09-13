"""车牌检测训练（YOLOv8s）。计划书 §6 M2。

车牌为小目标，训练输入分辨率默认 640（可调 1280 提升召回，见计划书 §7）。
⚠️ 需 GPU 环境执行。

用法：python tools/train_plate.py [--device 0] [--epochs 100] [--imgsz 640]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import CONFIGS_DIR, PROJECT_ROOT, WEIGHTS_DIR  # noqa: E402
from src.common.logger import setup_logger  # noqa: E402


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    setup_logger("train_plate")
    parser = argparse.ArgumentParser(description="训练车牌检测模型")
    parser.add_argument("--device", default="0", help="GPU 编号或 cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    logging.info("[Train] git commit=%s", _git_head())
    logging.info("[Train] 配置: device=%s epochs=%d batch=%d imgsz=%d",
                 args.device, args.epochs, args.batch, args.imgsz)

    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")
    results = model.train(
        data=str(CONFIGS_DIR / "plate_yolov8s.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
        patience=20,
        device=args.device,
        project=str(PROJECT_ROOT / "runs"),
        name="plate",
        seed=42,
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, WEIGHTS_DIR / "plate_best.pt")
        logging.info("[Train] 最佳权重已保存: %s", WEIGHTS_DIR / "plate_best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
