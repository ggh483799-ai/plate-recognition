"""数据集切分脚本：按「车牌号 hash」分组切分，防止同一辆车的数据泄漏到多个子集。

设计要点（对应计划书 §4.2 硬约束）：
- 同一车牌号的所有图片必须落在同一子集（train / val / test 之一）
- 分组 key 用稳定 hash 分配，保证切分可复现（跑两次结果一致）
- 同时适配两类数据：
  * 车牌识别数据（文件名含车牌号，如 `京A12345_xxx.jpg`）→ 按车牌号分组
  * 车辆检测数据（文件名不含车牌号）          → 按文件名一图一组

用法：
  python tools/split_dataset.py --src data/raw/ccpd      --dst data/splits
  python tools/split_dataset.py --check                  # 只校验，不切分
"""

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.logger import setup_logger  # noqa: E402

# ===== 常量 =====
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 中国车牌正则：省份简称 + 字母 + 5~6 位（蓝牌 7 位 / 绿牌 8 位）
_PLATE_RE = re.compile(
    r"([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]"
    r"[A-Z][A-HJ-NP-Z0-9]{5,6})"
)

# 切分比例与 hash 桶边界（同 key 永不跨子集）
SPLIT_NAMES = ("train", "val", "test")
_BUCKET_BOUNDS = {"train": 70, "val": 85, "test": 100}  # 70 / 15 / 15


def extract_plate(filename: str) -> str:
    """从文件名提取车牌号；提取不到返回空串。"""
    m = _PLATE_RE.search(filename)
    return m.group(1).upper() if m else ""


def group_key(filename: str) -> str:
    """返回分组 key：能提取车牌号则按车牌号分组，否则按文件名（一图一组）。"""
    plate = extract_plate(filename)
    return plate if plate else filename


def scan_images(src: Path) -> list:
    """递归扫描 src 下所有受支持的图片，返回按路径排序的列表。"""
    images = [p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(images)


def assign_split(key: str) -> str:
    """按 key 的稳定 hash 分配到 train/val/test，保证同 key 不跨子集、可复现。"""
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < _BUCKET_BOUNDS["train"]:
        return "train"
    if bucket < _BUCKET_BOUNDS["val"]:
        return "val"
    return "test"


def _print_summary(split_lists: dict) -> None:
    """打印各子集数量统计。"""
    total = sum(len(v) for v in split_lists.values())
    logging.info("==== 切分统计 ====")
    for name in SPLIT_NAMES:
        n = len(split_lists[name])
        pct = n / total * 100 if total else 0.0
        logging.info("  %-5s: %d 张 (%.1f%%)", name, n, pct)
    logging.info("  合计: %d 张", total)


def do_split(src: Path, dst: Path) -> int:
    """执行切分：扫描图片 -> 按车牌号分组 -> hash 分配 -> 写 txt。"""
    if not src.is_dir():
        logging.error("源目录不存在: %s", src)
        return 1

    images = scan_images(src)
    logging.info("扫描到 %d 张图片: src=%s", len(images), src)
    if not images:
        logging.warning("未发现图片，请检查 --src 路径与扩展名")
        return 1

    # 按分组 key 归类（同车牌号的图归一组）
    groups: dict = {}
    for p in images:
        groups.setdefault(group_key(p.name), []).append(p)
    logging.info("分组完成: %d 组（%d 张图片）", len(groups), len(images))

    # 分配子集
    split_lists = {name: [] for name in SPLIT_NAMES}
    for key, paths in groups.items():
        split_lists[assign_split(key)].extend(str(p) for p in paths)

    # 写 txt
    dst.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        out = dst / f"{name}.txt"
        out.write_text("\n".join(split_lists[name]) + "\n", encoding="utf-8")
        logging.info("[%s] 写入 %d 条: %s", name, len(split_lists[name]), out)

    _print_summary(split_lists)
    return 0


def do_check(dst: Path) -> int:
    """校验切分结果：数量统计 + 检查车牌号交叉（数据泄漏）。"""
    split_lists: dict = {}
    key_to_split: dict = {}
    overlap: list = []

    for name in SPLIT_NAMES:
        f = dst / f"{name}.txt"
        if not f.exists():
            logging.error("缺少切分文件: %s", f)
            return 1
        lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        split_lists[name] = lines
        for line in lines:
            key = group_key(Path(line).name)
            prev = key_to_split.get(key)
            if prev is not None and prev != name:
                overlap.append((key, prev, name))
            key_to_split[key] = name

    _print_summary(split_lists)
    if overlap:
        logging.error("发现 %d 个车牌号跨子集（数据泄漏）:", len(overlap))
        for key, a, b in overlap[:10]:
            logging.error("  车牌 %s 同时出现在 %s 和 %s", key, a, b)
        return 1
    logging.info("校验通过：无车牌号交叉")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="按车牌号 hash 分组切分数据集")
    parser.add_argument("--src", default="data/raw", help="原始图片目录")
    parser.add_argument("--dst", default="data/splits", help="切分文件输出目录")
    parser.add_argument("--check", action="store_true", help="校验已生成的切分结果")
    args = parser.parse_args(argv)

    setup_logger("split_dataset")
    if args.check:
        return do_check(Path(args.dst))
    return do_split(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    sys.exit(main())
