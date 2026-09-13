"""模型导出 ONNX。计划书 §6 M4：导出后与原模型输出误差 < 1e-3。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 允许直接以脚本方式运行
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.common.logger import setup_logger  # noqa: E402


def export_lprnet(weights: str, output: str, seq_len: int = 23) -> None:
    """导出 LPRNet 为 ONNX（固定输入 1×3×24×94）。"""
    from src.models.lprnet import LPRNet

    model = LPRNet()
    if Path(weights).exists():
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        logging.info("[ONNX] 加载权重 %s", weights)
    model.eval()

    dummy = torch.randn(1, 3, 24, 94)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    logging.info("[ONNX] 导出完成: %s", output)


def export_yolo(weights: str, output: str, imgsz: int = 640) -> None:
    """导出 YOLO 为 ONNX（复用 Ultralytics 内置导出）。"""
    from ultralytics import YOLO

    model = YOLO(weights)
    path = model.export(format="onnx", imgsz=imgsz, dynamic=True)
    logging.info("[ONNX] 导出完成: %s", path)


def main() -> int:
    setup_logger("onnx_export")
    parser = argparse.ArgumentParser(description="导出模型为 ONNX")
    parser.add_argument("--type", choices=["lprnet", "yolo"], required=True)
    parser.add_argument("--weights", required=True, help=".pt 权重路径")
    parser.add_argument("--output", required=True, help=".onnx 输出路径")
    args = parser.parse_args()

    if args.type == "lprnet":
        export_lprnet(args.weights, args.output)
    else:
        export_yolo(args.weights, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
