"""端到端延迟 / 吞吐压测。计划书 §6 M4：FPS、P50/P99 延迟。

用法：
    python tools/bench.py --img demo.jpg --n 50
    python tools/bench.py --img demo.jpg --concurrency 4 --n 100
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.common.logger import setup_logger  # noqa: E402


def bench_single(pipeline, img, n: int) -> dict:
    """单并发测单帧延迟，统计 P50/P99/均值。"""
    latencies: list[float] = []
    # 预热
    pipeline.run(img)
    for _ in range(n):
        t0 = time.perf_counter()
        pipeline.run(img)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    p50 = statistics.median(latencies)
    p99 = latencies[int(len(latencies) * 0.99)]
    fps = 1000.0 / (sum(latencies) / len(latencies)) if latencies else 0.0
    return {"n": n, "p50_ms": round(p50, 2), "p99_ms": round(p99, 2), "fps": round(fps, 2)}


def bench_concurrent(pipeline, img, n: int, concurrency: int) -> dict:
    """多线程并发压测，统计总吞吐。"""
    def _worker(_):
        pipeline.run(img)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(_worker, range(n)))
    elapsed = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "total_frames": n,
        "elapsed_s": round(elapsed, 2),
        "throughput_fps": round(n / elapsed, 2),
    }


def main() -> int:
    setup_logger("bench")
    parser = argparse.ArgumentParser(description="车牌识别端到端压测")
    parser.add_argument("--img", required=True, help="测试图片路径")
    parser.add_argument("--n", type=int, default=50, help="帧数")
    parser.add_argument("--concurrency", type=int, default=1, help="并发数")
    args = parser.parse_args()

    from src.io.reader import imread_bgr
    from src.models.detector import YoloDetector
    from src.models.lpr_recognizer import LprNetRecognizer
    from src.pipeline.lpr_pipeline import LprPipeline

    img = imread_bgr(args.img)
    if img is None:
        logging.error("[Bench] 无法读取图片: %s", args.img)
        return 1

    vehicle = YoloDetector("yolov8n.pt")
    plate = YoloDetector("yolov8n.pt")
    recognizer = LprNetRecognizer(None)
    pipeline = LprPipeline(vehicle, plate, recognizer)

    if args.concurrency > 1:
        result = bench_concurrent(pipeline, img, args.n, args.concurrency)
    else:
        result = bench_single(pipeline, img, args.n)

    logging.info("[Bench] 压测结果: %s", result)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
