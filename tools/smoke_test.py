"""部署冒烟测试：在容器内真实识别一次样图，证明"模型可用"而不只是"进程活着"。

healthz 200 测不出权重缺失 / 依赖损坏；这条脚本用 data/field/test_car.png
走一次 /predict_base64 全链路，断言读出预期车牌（粤A3333G）。
用法：sudo docker exec plate-recognition python tools/smoke_test.py
"""
from __future__ import annotations

import base64
import sys
import urllib.request
from pathlib import Path

EXPECTED_PLATE = "粤A3333G"
SAMPLE = Path("/app/data/field/test_car.png")
API = "http://127.0.0.1:8000/predict_base64"


def main() -> int:
    if not SAMPLE.is_file():
        print(f"[FAIL] 样图不存在: {SAMPLE}")
        return 1

    b64 = base64.b64encode(SAMPLE.read_bytes()).decode()
    req = urllib.request.Request(
        API,
        data=b'{"image_b64": "%s"}' % b64,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = __import__("json").loads(resp.read())
    except Exception as exc:
        print(f"[FAIL] 识别接口调用失败: {exc}")
        return 1

    plates = body.get("data", {}).get("plates", [])
    if not plates:
        print("[FAIL] 未识别出任何车牌（模型可能未正确加载）")
        return 1

    best = plates[0]
    no = best.get("plate_no", "")
    score = best.get("rec_score", 0)
    if EXPECTED_PLATE not in no or score < 0.9:
        print(f"[FAIL] 识别结果不符: {no} (rec={score})，期望包含 {EXPECTED_PLATE}")
        return 1

    print(f"[OK] 冒烟通过: {no} rec={score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
