"""CCPD 数据集解析脚本：把「文件名即标注」转成 YOLO 检测标注 + 车牌号清单。

CCPD 文件名 7 段（以 "-" 分隔），示例：
  025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
  ①区域比例 ②倾斜角 ③bbox左上&右下 ④四角点(右下起顺时针) ⑤车牌号索引 ⑥亮度 ⑦模糊度

字符映射（源自 CCPD 官方 README）：provinces / alphabets / ads 三个数组。

用法：
  python tools/prepare_ccpd.py --src data/raw/ccpd --out data/annotations/ccpd
  python tools/prepare_ccpd.py --src data/raw/ccpd --out data/annotations/ccpd --plate-csv data/ccpd_plates.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.logger import setup_logger  # noqa: E402

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# CCPD 标准图片尺寸（宽 x 高）
IMG_W, IMG_H = 1160, 720

# ---- CCPD 官方字符映射（来自 README） ----
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京",
             "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
             "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
             'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q',
       'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
       '6', '7', '8', '9', 'O']


def parse_filename(name: str):
    """解析文件名，返回 (x1, y1, x2, y2, plate)；无法解析返回 None。"""
    fields = Path(name).stem.split("-")
    if len(fields) < 6:
        return None
    bbox_field, plate_field = fields[2], fields[4]
    lu, rb = bbox_field.split("_")
    x1, y1 = (int(v) for v in lu.split("&"))
    x2, y2 = (int(v) for v in rb.split("&"))
    plate = decode_plate(plate_field)
    return x1, y1, x2, y2, plate


def decode_plate(field: str) -> str:
    """把车牌索引段解码为车牌号字符串（兼容 7 位蓝牌 / 8 位绿牌）。"""
    idxs = [int(x) for x in field.split("_")]
    chars = [PROVINCES[idxs[0]], ALPHABETS[idxs[1]]]
    chars += [ADS[i] for i in idxs[2:]]
    return "".join(chars)


def to_yolo(x1: int, y1: int, x2: int, y2: int):
    """xyxy 转 YOLO 归一化 cx,cy,w,h。"""
    cx = (x1 + x2) / 2 / IMG_W
    cy = (y1 + y2) / 2 / IMG_H
    w = (x2 - x1) / IMG_W
    h = (y2 - y1) / IMG_H
    return cx, cy, w, h


def prepare(src: Path, out: Path, plate_csv) -> int:
    """主流程：遍历图片 -> 解析 -> 写 YOLO txt（+ 可选车牌 CSV）。"""
    images = sorted([p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        logging.error("未发现图片: %s", src)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    ok = 0
    for p in images:
        parsed = parse_filename(p.name)
        if parsed is None:
            logging.warning("跳过（文件名无法解析）: %s", p.name)
            continue
        x1, y1, x2, y2, plate = parsed
        cx, cy, w, h = to_yolo(x1, y1, x2, y2)
        # 单类 plate，class_id=0
        (out / (p.stem + ".txt")).write_text(
            f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8"
        )
        rows.append((p.name, plate))
        ok += 1

    logging.info("解析完成: %d / %d 张", ok, len(images))
    if plate_csv:
        _write_plate_csv(plate_csv, rows)
    return 0


def _write_plate_csv(path: str, rows) -> None:
    """把 (文件名, 车牌号) 写入 CSV。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "plate"])
        w.writerows(rows)
    logging.info("车牌号清单已写入: %s（%d 条）", path, len(rows))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CCPD 文件名标注 -> YOLO 检测标注")
    parser.add_argument("--src", required=True, help="CCPD 图片目录")
    parser.add_argument("--out", required=True, help="YOLO 标注输出目录")
    parser.add_argument("--plate-csv", default=None, help="可选：车牌号 CSV 输出路径")
    args = parser.parse_args(argv)

    setup_logger("prepare_ccpd")
    src = Path(args.src)
    if not src.is_dir():
        logging.error("源目录不存在: %s", src)
        return 1
    return prepare(src, Path(args.out), args.plate_csv)


if __name__ == "__main__":
    sys.exit(main())
