"""端到端验证浏览器摄像头链路：模拟浏览器「取帧 → 推帧 → 收结果」全过程。

为什么要有这个脚本
----------------
页面上的摄像头采集（getUserMedia）只能在**真实浏览器**里跑，自动化沙盒点不出来
（agent-browser 在本环境会挂死）。但它推给服务端的东西是确定的：**一张 JPEG + 会话号**。
本脚本就用系统里的真实视频逐帧编码成 JPEG 推上去，把服务端这一侧完整跑一遍：
建会话 → 推帧 → 拿框/事件/统计 → 取截图 → 收尾。

用法：python tools/verify_camera_api.py [视频文件或图片] [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import requests  # noqa: E402

from src.io.reader import imread_bgr  # noqa: E402  —— 中文路径必须走它（cv2.imread 会返回 None）

OUT = ROOT / "runs" / "camera_e2e"


def frames_from(path: pathlib.Path, limit: int, max_side: int):
    """把素材变成一串 JPEG 字节（就是浏览器 canvas.toBlob 的输出）。"""
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = imread_bgr(path)          # 中文路径安全
        if img is None or not getattr(img, "size", 0):
            raise SystemExit(f"读不出图片: {path}")
        for _ in range(limit):
            yield _encode(img, max_side)
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {path}")
    try:
        sent = 0
        while sent < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield _encode(frame, max_side)
            sent += 1
    finally:
        cap.release()


def _encode(frame, max_side: int) -> bytes:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        raise SystemExit("JPEG 编码失败")
    return buf.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=str(ROOT / "data" / "field" / "test_car.png"))
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--interval-ms", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--level", default="low")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    s = requests.Session()

    print(f"[1] 建会话（模拟浏览器点「开始实时识别」）base={base}")
    r = s.post(f"{base}/camera/session",
               data={"interval_ms": args.interval_ms, "level": args.level,
                     "max_side": args.max_side, "min_det": 0.3, "min_rec": 0.35},
               timeout=180)
    print(f"    HTTP {r.status_code} {str(r.json())[:150]}")
    if r.status_code != 200:
        return 1
    d = r.json()["data"]
    sid, frame_url = d["session_id"], d["frame_url"]
    print(f"    session={sid} frame_url={frame_url} interval={d['interval_ms']}ms "
          f"镜像上限={d['max_frame_bytes'] // 1048576}MB")

    print(f"[2] 逐帧推流（源={pathlib.Path(args.source).name}）")
    boxes_seen, last = 0, None
    t0 = time.time()
    for i, jpeg in enumerate(frames_from(pathlib.Path(args.source), args.frames, args.max_side), 1):
        rr = s.post(f"{base}{frame_url}", files={"file": ("frame.jpg", jpeg, "image/jpeg")}, timeout=120)
        if rr.status_code != 200:
            print(f"    第 {i} 帧 HTTP {rr.status_code}: {rr.text[:120]}")
            ok = False
            break
        last = rr.json()["data"]
        boxes_seen = max(boxes_seen, len(last["boxes"]))
        if i <= 3 or i == args.frames:
            kinds = [b["kind"] for b in last["boxes"]]
            plate = [b["label"] for b in last["boxes"] if b["kind"] == "plate"]
            print(f"    第 {i:2d} 帧: 框 {len(last['boxes'])} 个 {kinds} 车牌={plate} "
                  f"reused={last['reused']} 识别耗时={last['stats']['avg_recog_ms']}ms")
    elapsed = time.time() - t0
    if last is None:
        print("    没有任何帧成功推上去")
        return 1
    print(f"    推完耗时 {elapsed:.1f}s（平均 {elapsed / max(1, args.frames) * 1000:.0f}ms/帧）")

    st = last["stats"]
    print(f"[3] 统计: frames={st['frames']} recog={st['recog_runs']} dropped={st['recog_dropped']} "
          f"skipped={st['skipped']} push_fps={st['push_fps']} recog_fps={st['recog_fps']} "
          f"backend={st['backend']}")
    print(f"    最大同屏框数={boxes_seen} 事件={st['events']} 命中={st['total_hits']}")

    print("[4] 事件与截图")
    for ev in last["events"][:5]:
        print(f"    {ev['plate_no']} / {ev['plate_color']} / {ev['vehicle_type']} "
              f"det={ev['det_score']:.3f} rec={ev['rec_score']:.3f} 命中={ev['hits']}")
    if last["media_base"] and last["events"]:
        shot = last["events"][0].get("shot")
        url = f"{base}{last['media_base']}/{shot}"
        rr = s.get(url, timeout=30)
        print(f"    截图 {url} → HTTP {rr.status_code} {len(rr.content)}B "
              f"jpeg={rr.content[:2] == b'\xff\xd8'}")
        if rr.status_code != 200 or rr.content[:2] != b"\xff\xd8":
            ok = False
        else:
            (OUT / shot).write_bytes(rr.content)

    print("[5] 状态查询 + 收尾")
    snap = s.get(f"{base}/camera/{sid}", timeout=30).json()["data"]
    print(f"    GET /camera/{sid} → status={snap['status']} events={len(snap['events'])}")
    if last["events"]:
        (OUT / "boxes.txt").write_text(
            "\n".join(f"{b['kind']}\t{b['label']}\t{b['score']}\t{b['bbox']}" for b in last["boxes"]),
            encoding="utf-8")
    print(f"    DELETE → {s.delete(f'{base}/camera/{sid}', params={'purge': 'true'}, timeout=60).status_code}")
    print("[6] 结论:", "PASS" if ok and st["recog_runs"] >= 1 else "FAIL")
    return 0 if (ok and st["recog_runs"] >= 1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
