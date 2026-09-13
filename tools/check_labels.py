"""YOLO 标注质检脚本：检查越界框、空标签、class_id 越界、格式错误。

验收（M1）：python tools/check_labels.py --labels data/annotations --imgs data/raw/ccpd
通过标准：无越界框、无空标签。

用法：
  python tools/check_labels.py --labels data/annotations                    # 只查标注文件
  python tools/check_labels.py --labels data/annotations --imgs data/raw/ccpd   # 额外查图-标注对应
  python tools/check_labels.py --labels data/annotations --classes vehicle  # 单类别（默认）
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.logger import setup_logger  # noqa: E402

# ===== 常量 =====
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_label_line(line: str, lineno: int):
    """解析一行标注，返回 (class_id, cx, cy, w, h)；格式错误抛 ValueError。"""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"列数错误: {len(parts)}（应为 5）")
    vals = [float(x) for x in parts]
    cls = int(vals[0])
    if vals[0] != float(cls) or cls < 0:
        raise ValueError(f"class_id 非法: {parts[0]}")
    return cls, vals[1], vals[2], vals[3], vals[4]


def bbox_valid(cx: float, cy: float, w: float, h: float) -> bool:
    """判断归一化 bbox 是否合法（坐标在 [0,1] 且框体不越界、宽高为正）。"""
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        return False
    if w <= 0 or h <= 0:
        return False
    if cx - w / 2 < 0 or cx + w / 2 > 1 or cy - h / 2 < 0 or cy + h / 2 > 1:
        return False
    return True


def check_labels(labels_dir: Path, imgs_dir, n_classes: int) -> int:
    """质检主流程：逐文件解析，累计各类错误，返回错误条数。"""
    label_files = sorted(labels_dir.rglob("*.txt"))
    if not label_files:
        logging.error("未发现标注文件: %s", labels_dir)
        return 1

    errors = {"越界框": [], "空标签": [], "class_id越界": [], "格式错误": []}
    n_boxes = 0
    for f in label_files:
        lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            errors["空标签"].append(str(f))
            continue
        for i, ln in enumerate(lines, start=1):
            try:
                cls, cx, cy, w, h = parse_label_line(ln, i)
            except ValueError as e:
                errors["格式错误"].append(f"{f}:{i} -> {e}")
                continue
            if cls >= n_classes:
                errors["class_id越界"].append(f"{f}:{i} -> class_id={cls}（类别数 {n_classes}）")
            if not bbox_valid(cx, cy, w, h):
                errors["越界框"].append(f"{f}:{i} -> {cx:.3f},{cy:.3f},{w:.3f},{h:.3f}")
            n_boxes += 1

    logging.info("==== 标注质检统计 ====")
    logging.info("  标注文件: %d 个", len(label_files))
    logging.info("  标注框总数: %d", n_boxes)
    total_err = 0
    for name, items in errors.items():
        if items:
            logging.warning("  %s: %d 处", name, len(items))
            for it in items[:5]:
                logging.warning("    %s", it)
        total_err += len(items)

    _check_image_pairing(imgs_dir, label_files)
    if total_err == 0:
        logging.info("质检通过：无越界框、无空标签、无格式错误")
        return 0
    logging.error("质检未通过：共 %d 处问题", total_err)
    return 1


def _check_image_pairing(imgs_dir, label_files) -> None:
    """检查图片与标注是否一一对应（有图无标注 / 有标注无图）。"""
    if not imgs_dir or not imgs_dir.is_dir():
        return
    imgs = {p.stem for p in imgs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS}
    label_stems = {p.stem for p in label_files}
    missing = sorted(imgs - label_stems)
    orphan = sorted(label_stems - imgs)
    logging.info("  图片: %d 张, 标注: %d 份", len(imgs), len(label_stems))
    if missing:
        logging.warning("  有图无标注: %d 张（前 5）: %s", len(missing), missing[:5])
    if orphan:
        logging.warning("  有标注无图: %d 份（前 5）: %s", len(orphan), orphan[:5])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="YOLO 标注质检")
    parser.add_argument("--labels", required=True, help="标注目录（含 .txt）")
    parser.add_argument("--imgs", default=None, help="图片目录（可选，查图-标注对应）")
    parser.add_argument("--classes", nargs="+", default=["vehicle"], help="类别名列表")
    args = parser.parse_args(argv)

    setup_logger("check_labels")
    labels_dir = Path(args.labels)
    if not labels_dir.is_dir():
        logging.error("标注目录不存在: %s", labels_dir)
        return 1
    return check_labels(labels_dir, Path(args.imgs) if args.imgs else None, len(args.classes))


if __name__ == "__main__":
    sys.exit(main())
