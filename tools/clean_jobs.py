"""清理任务产物目录（默认同时清理视频任务与实时会话两类）。

目录
----
    runs/jobs/<job_id>/        视频异步任务：input / annotated.mp4 / events.csv / shots/
    runs/streams/<sid>/        实时会话（服务端拉流）：input / shots/
    runs/cameras/<sid>/        浏览器摄像头会话：shots/

为什么单独给个脚本
------------------
1. 服务端产物默认保留一段时间（`service/jobs.py` TTL 6h、`service/streams.py` TTL 30min），
   服务重启时会自动清一次；但**装了删除钩子的受限环境**会拦住服务的批量删除，
   此时接口会如实上报 `purged=false` 而不是假装删掉——需要人工跑一次本脚本
   （用户自己的终端通常没有钩子）。
2. 产物比图片大得多（标注视频 + 每个车牌一张小图），跑几段视频就能占掉几百 MB。

用法
----
    python tools/clean_jobs.py                 # 预览：列出所有任务目录与占用
    python tools/clean_jobs.py --all --yes     # 删除全部
    python tools/clean_jobs.py --ttl 6 --yes   # 只删结束超过 6 小时的（按目录 mtime 估算）
    python tools/clean_jobs.py --root runs/jobs --all --yes     # 只清某一类（可逗号分隔多个）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.logger import setup_logger  # noqa: E402
from service.jobs import _remove_tree  # noqa: E402

DEFAULT_ROOTS = ("runs/jobs", "runs/streams", "runs/cameras")


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _fmt(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1048576:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1048576:.1f} MB"


def _collect(root: Path, ttl_hours: float) -> list[tuple[Path, int, float]]:
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        age_h = (time.time() - d.stat().st_mtime) / 3600.0
        if ttl_hours > 0 and age_h < ttl_hours:
            continue
        out.append((d, _dir_size(d), age_h))
    return out


def main() -> int:
    setup_logger("clean_jobs")
    parser = argparse.ArgumentParser(description="清理视频任务 / 实时会话的产物目录")
    parser.add_argument("--root", default="",
                        help=f"要清理的目录，逗号分隔；默认 {' 与 '.join(DEFAULT_ROOTS)}")
    parser.add_argument("--all", action="store_true", help="删除匹配到的全部任务目录")
    parser.add_argument("--ttl", type=float, default=0.0,
                        help="只删最后修改超过 N 小时的任务目录")
    parser.add_argument("--yes", action="store_true", help="跳过确认提示")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    roots = ([Path(x) for x in args.root.split(",") if x.strip()] if args.root
             else [base / r for r in DEFAULT_ROOTS])

    # 危险操作先列清单（全局规则 4.2）
    targets: list[tuple[Path, int, float]] = []
    for root in roots:
        targets.extend(_collect(root, args.ttl))

    if not targets:
        print(f"[INFO] 没有匹配的任务目录（{', '.join(str(r) for r in roots)}）")
        return 0

    print(f"将删除以下 {len(targets)} 个任务目录：")
    total = 0
    for d, size, age_h in targets:
        total += size
        print(f"  - {d}  {_fmt(size):>9}  最后修改 {age_h:.1f} 小时前")
    print(f"合计占用 {_fmt(total)}")

    if not (args.all or args.ttl > 0):
        print("[INFO] 仅预览。加 --all 或 --ttl N 才会真正删除。")
        return 0
    if not args.yes:
        print("[INFO] 为安全起见需要显式 --yes 才执行删除。")
        return 1

    purged = failed = 0
    for d, _size, _age in targets:
        if _remove_tree(d):
            purged += 1
        else:
            failed += 1
    print(f"[INFO] 已删除 {purged} 个；失败 {failed} 个"
          + ("（环境删不掉，请手动处理）" if failed else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
