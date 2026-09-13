"""视频识别接口端到端验证（走真实模型，非 fake）。

用法：python tools/verify_video_api.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import requests

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "video_api_e2e"
OUT.mkdir(parents=True, exist_ok=True)
VIDEO = ROOT / "service" / "static" / "demo.mp4"


def post_video(path: Path, name: str, **form):
    with open(path, "rb") as fh:
        return requests.post(
            f"{BASE}/predict_video",
            files={"file": (name, fh, "video/mp4")},
            data={"window_s": str(form.get("window_s", 3.0)),
                  "max_frames": str(form.get("max_frames", 1200))},
            timeout=120,
        )


def main() -> int:
    print("=" * 68)
    print(f"[1] 上传 {VIDEO.name} ({VIDEO.stat().st_size / 1024:.0f} KB)")
    t0 = time.time()
    r = post_video(VIDEO, VIDEO.name)
    print("    HTTP", r.status_code, r.text[:200])
    if r.status_code != 200:
        return 1
    start = r.json()["data"]
    job_id = start["job_id"]
    print(f"    job_id={job_id} status={start['status']} max_frames={start['max_frames']}")

    print("[2] 轮询")
    last = None
    while True:
        s = requests.get(f"{BASE}/predict_video/{job_id}", timeout=30).json()["data"]
        p = s["progress"]
        line = f"    status={s['status']:<8} {p['frames']}/{p['total_frames']} 帧 ({p['percent']}%) 已用 {p['elapsed_s']}s eta={p['eta_s']}"
        if line != last:
            print(line)
            last = line
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.8)

    if s["status"] != "done":
        print("    失败:", s["error"])
        return 1

    print(f"[3] 完成，墙钟 {time.time() - t0:.1f}s")
    print("    stats :", json.dumps(s["stats"], ensure_ascii=False))
    print("    urls  :", json.dumps(s["urls"], ensure_ascii=False))
    print("    事件  :")

    ok = True
    for ev in s["events"]:
        print("      -", json.dumps(ev, ensure_ascii=False))

    if not s["events"]:
        print("    !! 没有识别到车牌")
        ok = False
    else:
        ev = s["events"][0]
        if ev["hits"] < 5:
            print(f"    !! 命中帧数偏低: {ev['hits']}")
            ok = False

    print("[4] 拉取产物并校验")
    files = {"annotated.mp4": "annotated", "events.csv": "csv", "events.jsonl": "jsonl"}
    for name, key in files.items():
        url = s["urls"][key]
        if not url:
            print(f"    !! {name} 无 URL")
            ok = False
            continue
        data = requests.get(BASE + url, timeout=60).content
        (OUT / name).write_bytes(data)
        print(f"    {name}: {len(data)} B -> {OUT / name}")

    csv_bytes = (OUT / "events.csv").read_bytes()
    if not csv_bytes.startswith(b"\xef\xbb\xbf"):
        print("    !! CSV 缺 BOM")
        ok = False
    else:
        print("    CSV BOM = EF BB BF ✓")

    shots_url = s["urls"]["shots"]
    for ev in s["events"]:
        if not ev["shot"]:
            continue
        url = f"{shots_url}/{ev['shot']}"
        data = requests.get(BASE + url, timeout=30).content
        (OUT / ev["shot"]).write_bytes(data)
        magic = data[:2] == b"\xff\xd8"
        print(f"    {ev['shot']}: {len(data)} B jpeg={magic} -> {OUT / ev['shot']}")
        if not magic or len(data) < 200:
            ok = False

    cap = cv2.VideoCapture(str(OUT / "annotated.mp4"))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    print(f"    标注视频回读 {decoded} 帧 / 统计声称 {s['stats']['frames']} 帧")
    if decoded != s["stats"]["frames"]:
        print("    !! 标注视频帧数与统计不一致（静默丢帧）")
        ok = False

    print("[5] 单图批量接口")
    b = requests.post(
        f"{BASE}/predict_batch",
        files=[("files", ("a.jpg", open(ROOT / "service/static/demo.jpg", "rb"), "image/jpeg")),
               ("files", ("bad.jpg", b"not-an-image", "image/jpeg"))],
        timeout=120,
    )
    bd = b.json()["data"]
    print("    count=%d ok=%d plates=%d elapsed=%.0fms" % (
        bd["count"], bd["ok_count"], bd["total_plates"], bd["elapsed_ms"]))
    for it in bd["items"]:
        print("      -", it["name"], "plates=%d" % len(it["plates"]), "error=%r" % it["error"])
    if bd["count"] != 2 or bd["ok_count"] != 1 or bd["total_plates"] < 1:
        print("    !! 批量结果不符合预期")
        ok = False

    print("[6] 任务清理")
    d = requests.delete(f"{BASE}/predict_video/{job_id}", timeout=30)
    print("    DELETE ->", d.status_code, d.text[:120])

    print("=" * 68)
    print("结论:", "全部通过 ✅" if ok else "存在失败项 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
