"""COCO 车辆子集下载脚本：用 fiftyone 下载 car/bus/truck/motorcycle 并导出 YOLO。

依赖：pip install fiftyone（已列入 requirements.txt）

用法：
  python tools/download_coco.py --split train --max 5000 --out data/raw/coco_vehicle
  python tools/download_coco.py --split validation --max 1000 --out data/raw/coco_vehicle_val
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.logger import setup_logger  # noqa: E402

COCO_VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]


def download(split: str, max_samples, out: Path) -> int:
    """下载 COCO 车辆子集并导出 YOLOv5 格式。"""
    import fiftyone as fo
    import fiftyone.zoo as foz

    logging.info("下载 COCO-2017 %s 车辆子集（max=%s）", split, max_samples)
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=COCO_VEHICLE_CLASSES,
        max_samples=max_samples,
    )
    logging.info("样本数: %d，导出 YOLOv5 格式到 %s", len(dataset), out)
    dataset.export(
        export_dir=str(out),
        dataset_type=fo.types.YOLOv5Dataset,
        classes=COCO_VEHICLE_CLASSES,
    )
    logging.info("导出完成: %s", out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="COCO 车辆子集下载")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max", type=int, default=None, help="最大样本数（None=全部）")
    parser.add_argument("--out", required=True, help="输出目录")
    args = parser.parse_args(argv)

    setup_logger("download_coco")
    try:
        return download(args.split, args.max, Path(args.out))
    except ImportError:
        logging.error("缺少 fiftyone，请先: pip install fiftyone")
        return 1


if __name__ == "__main__":
    sys.exit(main())
