"""端到端验证：网页视频链接（微博/抖音/B站…）→ 解析直链 → 实时拉流识别。

用法：python tools/verify_online_link.py [链接]（默认为一条微博视频）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

BASE = "http://127.0.0.1:8000"
WEIBO_URL = sys.argv[1] if len(sys.argv) > 1 else "https://weibo.com/tv/show/1034:5156125327425586"
OUT = Path(__file__).resolve().parents[1] / "runs" / "stream_online"
OUT.mkdir(parents=True, exist_ok=True)


def mjpeg_frame(url: str, timeout: float = 8.0) -> bytes | None:
    """从 MJPEG 流里取一帧 JPEG。"""
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        buf = b""
        t0 = time.time()
        for chunk in r.iter_content(8192):
            buf += chunk
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
            if s >= 0 and e > 0:
                return buf[s:e + 2]
            if time.time() - t0 > timeout:
                break
    return None


def main() -> bool:
    t0 = time.time()
    r = requests.post(BASE + "/stream", data={
        "source": WEIBO_URL, "interval_ms": "500", "level": "low",
        "max_side": "640", "min_det": "0.3", "min_rec": "0.35",
    }, timeout=120)
    print("[1] POST /stream ->", r.status_code)
    if r.status_code != 200:
        print(r.text[:400])
        return False
    sid = r.json()["data"]["session_id"]
    mjpg = BASE + f"/stream/{sid}.mjpg"

    last = ""
    got_frame = None
    while time.time() - t0 < 90:
        st = requests.get(f"{BASE}/stream/{sid}", timeout=30).json()["data"]
        s = st["stats"]
        line = (f"t={time.time()-t0:5.1f}s {st['status']:8s} read={s['read_frames']:4d} "
                f"push={s['frames']:4d} recog={s['recog_runs']:3d} ev={s['events']:2d} "
                f"title={st.get('source_title','')[:24]!r} err={st['error'][:60]!r}")
        if line != last:
            print("   ", line, flush=True)
            last = line
        if st["status"] in ("failed", "stopped"):
            break
        if st["status"] == "running" and got_frame is None and s["frames"] > 0:
            got_frame = mjpeg_frame(mjpg)
            if got_frame:
                (OUT / "mjpeg_frame.jpg").write_bytes(got_frame)
                print(f"[2] MJPEG 首帧已保存: {len(got_frame)} B, jpeg={got_frame[:2] == b'\\xff\\xd8'}")
        if s["events"] >= 1 and s["frames"] > 100:
            break
        time.sleep(2)

    st = requests.get(f"{BASE}/stream/{sid}", timeout=30).json()["data"]
    s = st["stats"]
    print(f"[3] 最终: status={st['status']} push={s['frames']}帧/{s['elapsed_s']}s "
          f"(≈{s['push_fps']}fps) recog={s['recog_runs']}次/{s['avg_recog_ms']}ms "
          f"事件={s['events']} 命中={s['total_hits']}")
    for ev in st["events"][:6]:
        print(f"    {ev['plate_no']} / {ev['plate_color']} / {ev['vehicle_type']} "
              f"det={ev['det_score']:.4f} rec={ev['rec_score']:.4f} hits={ev['hits']}")
        if ev.get("shot") and st.get("media_base"):
            img = requests.get(f"{BASE}{st['media_base']}/{ev['shot']}", timeout=30)
            if img.status_code == 200 and img.content[:2] == b"\xff\xd8":
                (OUT / ev["shot"]).write_bytes(img.content)
                print(f"    截图已保存 {ev['shot']} ({len(img.content)} B)")
            else:
                print(f"    截图拉取失败 HTTP {img.status_code}")
                return False

    ok = (st["status"] in ("running", "stopped") and s["frames"] > 0)
    if got_frame is not None and got_frame[:2] == b"\xff\xd8":
        # 抓到的画面帧要真的有内容（不是黑屏/HTML）
        arr = cv2.imdecode(np.frombuffer(got_frame, np.uint8), cv2.IMREAD_COLOR)
        print(f"[4] 画面帧均值亮度={arr.mean():.1f}（<5 视为黑屏）")
        ok = ok and arr.mean() > 5
    requests.delete(f"{BASE}/stream/{sid}?purge=true", timeout=60)
    print("[5] 会话已清理, 验证结果:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
