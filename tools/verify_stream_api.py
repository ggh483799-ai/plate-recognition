"""实时流接口端到端验证（走真实模型，非 fake）。

用法：先启动服务，再跑
    python tools/verify_stream_api.py

覆盖：
  1. 本地视频源：启动会话 → 读 MJPEG → 校验画面确实被画过框 → 统计/事件/截图 → 停止
  2. 网络源（http:// 指向本服务的示例视频）：验证非本地文件来源也能跑（RTSP 同属远程源）
  3. 摄像头索引：有摄像头就读几帧；没有就应当**优雅失败**，而不是把服务搞崩

RTSP 真流需要真实设备或 mediamtx 推流，本脚本不覆盖——无源可测就如实说明，不假装通过。
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
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "stream_e2e"
OUT.mkdir(parents=True, exist_ok=True)
DEMO = ROOT / "service" / "static" / "demo.mp4"
BOUNDARY = b"--plateframeboundary"


def make_long_clip(src: Path, dst: Path, repeat: int = 3) -> Path:
    """把示例视频复读几遍凑一段更长的素材。

    示例只有 4 秒（32 帧），识别按 ~1 次/秒算只够 1~2 次，统计没有代表性；
    复读成 12 秒后识别次数、命中帧数才有意义。
    """
    if dst.is_file():
        return dst
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return src
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(repeat):
        for frame in frames:
            writer.write(frame)
    writer.release()
    print(f"    已生成 {dst.name}: {len(frames) * repeat} 帧 @ {fps:.0f}fps "
          f"（{len(frames) * repeat / fps:.0f}s）")
    return dst


def read_mjpeg(url: str, seconds: float = 30.0, max_frames: int = 500, read_timeout: float = 30.0):
    """读一段 MJPEG，返回 (帧列表, 原始字节数)。

    流结束、超时、对端长时间不给数据（设备无输出）都会正常返回已收到的部分，
    不向调用方抛异常——"读不到帧"本身就是需要观测的结果。
    """
    frames: list[bytes] = []
    total = 0
    deadline = time.time() + seconds
    try:
        with requests.get(url, stream=True, timeout=(10, read_timeout)) as resp:
            resp.raise_for_status()
            buf = b""
            for chunk in resp.iter_content(chunk_size=8192):
                buf += chunk
                total += len(chunk)
                while True:
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start < 0 or end < 0:
                        break
                    frames.append(buf[start:end + 2])
                    buf = buf[end + 2:]
                    if len(frames) >= max_frames:
                        return frames, total
                if time.time() > deadline:
                    break
    except requests.exceptions.RequestException as exc:
        print(f"    （读流中断：{type(exc).__name__}——这段时间没有拿到任何新帧）")
    return frames, total


def start_stream(**kwargs) -> dict:
    files = None
    data = {}
    for key, value in kwargs.items():
        if key == "file":
            files = {"file": (value[0], open(value[1], "rb"), "video/mp4")}
        else:
            data[key] = str(value)
    r = requests.post(f"{BASE}/stream", files=files, data=data, timeout=300)
    return {"status_code": r.status_code, "body": r.json() if r.content else {}}


def stop_stream(sid: str, purge: bool = True) -> dict:
    r = requests.delete(f"{BASE}/stream/{sid}", params={"purge": str(purge).lower()}, timeout=60)
    return {"status_code": r.status_code, "body": r.json() if r.content else {}}


def status_of(sid: str) -> dict:
    return requests.get(f"{BASE}/stream/{sid}", timeout=60).json()["data"]


def main() -> int:
    ok = True

    print("=" * 70)
    print("[1] 本地视频源：启动会话")
    clip = make_long_clip(DEMO, OUT / "demo_long.mp4", repeat=3)
    res = start_stream(file=(clip.name, clip), interval_ms=200, level="low", max_side=640)
    print("    HTTP", res["status_code"], json.dumps(res["body"], ensure_ascii=False)[:220])
    if res["status_code"] != 200:
        print("    !! 启动失败")
        return 1
    data = res["body"]["data"]
    sid = data["session_id"]
    print(f"    session={sid} mjpeg={data['mjpeg_url']} level={data['level']} size={data['max_side']}")

    print("[2] 读 MJPEG 推流")
    t0 = time.time()
    frames, total = read_mjpeg(BASE + data["mjpeg_url"], seconds=40.0)
    wall = time.time() - t0
    print(f"    收到 {len(frames)} 帧 / {total} B / 用时 {wall:.1f}s（≈{len(frames) / wall:.1f} fps）")
    if len(frames) < 20:
        print("    !! 推流帧数过少（画面被识别卡住了？）")
        ok = False

    if frames:
        for tag, jpeg in (("首", frames[0]), ("中", frames[len(frames) // 2]), ("末", frames[-1])):
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"    !! {tag}帧无法解码")
                ok = False
                continue
            print(f"    第 {tag} 帧: {img.shape[1]}x{img.shape[0]} {len(jpeg)} B")
        (OUT / "mjpeg_last.jpg").write_bytes(frames[-1])

        # 画面确实被"画过"：与源视频首帧对比应有差异（框 + HUD）
        cap = cv2.VideoCapture(str(clip))
        ok_src, src = cap.read()
        cap.release()
        first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
        if ok_src and first is not None:
            h = min(src.shape[0], first.shape[0])
            w = min(src.shape[1], first.shape[1])
            diff = int(np.abs(src[:h, :w].astype(int) - first[:h, :w].astype(int)).sum())
            print(f"    与源视频首帧的像素差 = {diff}（>0 说明框/HUD 已画上）")
            if diff <= 0:
                print("    !! 画面没有任何标注")
                ok = False

    print("[3] 会话状态")
    st = status_of(sid)
    s = st["stats"]
    print(f"    status={st['status']} error={st['error']!r}")
    print(f"    预热={s['warmup_ms']:.0f}ms  frames={s['frames']} read={s['read_frames']} "
          f"skipped={s['skipped']}")
    print(f"    recog_runs={s['recog_runs']} recog_dropped={s['recog_dropped']} "
          f"avg_recog_ms={s['avg_recog_ms']} push_fps={s['push_fps']} recog_fps={s['recog_fps']}")
    print(f"    events={s['events']} hits={s['total_hits']} live={s['live']}")
    for ev in st["events"]:
        print("      -", json.dumps(ev, ensure_ascii=False))

    # 关键指标：画面没有被识别拖住（早前同步实现里 push_fps 只有 0.7）
    if s["push_fps"] < 3.0:
        print(f"    !! 推送帧率过低 {s['push_fps']}——识别与播放没有解耦")
        ok = False
    if s["recog_runs"] < 2:
        print(f"    !! 识别次数过少 {s['recog_runs']}")
        ok = False
    if not st["events"]:
        print("    !! 没有识别到车牌")
        ok = False
    else:
        ev = st["events"][0]
        if ev["plate_no"] != "粤A3333G" or ev["hits"] < 2:
            print("    !! 车牌或命中帧数不符合预期")
            ok = False
        if not ev["shot"]:
            print("    !! 没有截图")
            ok = False
        else:
            url = f"{BASE}{st['media_base']}/{ev['shot']}"
            r = requests.get(url, timeout=30)
            # 先把字节比对算出来：直接写进 f-string 会被反斜杠转义绕晕（踩过一次）
            is_jpeg = r.content[:2] == b"\xff\xd8"
            print(f"    截图 {url} → HTTP {r.status_code}, {len(r.content)} B, jpeg={is_jpeg}")
            if r.status_code != 200 or not is_jpeg:
                ok = False
            (OUT / ev["shot"]).write_bytes(r.content)

    print("[4] 停止会话（保留记录）")
    r = requests.delete(f"{BASE}/stream/{sid}", timeout=60)
    print("    DELETE →", r.status_code, json.dumps(r.json(), ensure_ascii=False)[:160])
    st2 = status_of(sid)
    print(f"    停止后状态 = {st2['status']}（事件仍可读: {len(st2['events'])} 个）")
    if st2["status"] != "stopped":
        print("    !! 停止后状态不是 stopped")
        ok = False

    print("[5] 网络源（http:// 指向本服务的示例视频，代表「远程源」这类场景）")
    res2 = start_stream(source=f"{BASE}/static/demo.mp4", interval_ms=300, level="low", max_side=480)
    print("    HTTP", res2["status_code"], json.dumps(res2["body"], ensure_ascii=False)[:200])
    if res2["status_code"] == 200:
        sid2 = res2["body"]["data"]["session_id"]
        f2, t2 = read_mjpeg(BASE + res2["body"]["data"]["mjpeg_url"], seconds=25.0)
        st3 = status_of(sid2)
        print(f"    收到 {len(f2)} 帧 / {t2} B；status={st3['status']} events={st3['stats']['events']} "
              f"err={st3['error']!r}")
        if len(f2) < 2 and st3["status"] == "failed":
            print("    !! 网络源不可用")
            ok = False
        stop_stream(sid2, purge=True)
    else:
        print("    !! 网络源启动被拒")

    print("[6] 摄像头索引")
    res3 = start_stream(source="0", interval_ms=500, level="low", max_side=480)
    if res3["status_code"] == 200:
        sid3 = res3["body"]["data"]["session_id"]
        f3, _ = read_mjpeg(BASE + res3["body"]["data"]["mjpeg_url"], seconds=10.0)
        st4 = status_of(sid3)
        print(f"    status={st4['status']} error={st4['error']!r} "
              f"frames={st4['stats']['frames']} 收到={len(f3)} 帧")
        if st4["status"] == "failed":
            print("    ✓ 无摄像头时优雅失败，服务未受影响")
        elif len(f3) >= 1:
            print("    ✓ 摄像头可读（本机有可用视频设备），实时链路通")
        else:
            print("    （设备打开但没读到帧，可能是虚拟摄像头返回空流）")
        stop_stream(sid3, purge=True)
    else:
        print("    启动被拒:", res3["status_code"], res3["body"])

    print("[7] 清理本地视频会话产物")
    print("   ", json.dumps(stop_stream(sid, purge=True)["body"], ensure_ascii=False)[:200])

    print("=" * 70)
    print("结论:", "全部通过 ✅" if ok else "存在失败项 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
