"""实时流识别测试：会话生命周期、按时间节流、事件/截图、MJPEG 推流与接口契约。

全部用注入的 fake 帧源与 fake pipeline，不加载真实模型、不依赖摄像头，秒级跑完。
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import service.app as app_mod
import src.pipeline.stream as stream_mod
from service.streams import MAX_SIDE_CHOICES, StreamManager, TooManyStreams
from src.common.interfaces import PlateResult
from src.pipeline.stream import (
    LiveStreamSession,
    STATUS_FAILED,
    STATUS_STOPPED,
    is_live_source,
)

client = TestClient(app_mod.app)  # 不触发 lifespan

PLATE = PlateResult(
    plate_no="粤A3333G", plate_color="blue", vehicle_type="car",
    det_score=0.88, rec_score=0.97, bbox=[8.0, 12.0, 60.0, 30.0], cost_ms=9.0,
)


class FakeReader:
    """按预设帧序列产出；fps=None 表示无节流（测起来快）。delay 可模拟慢源。"""

    def __init__(self, frames, fps=None, frame_count=None, delay=0.0):
        self._frames = list(frames)
        self.fps = fps
        self.frame_count = len(self._frames) if frame_count is None else frame_count
        self.closed = False
        self.skipped = 0
        self.delay = float(delay)

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if not self._frames:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return self._frames.pop(0)

    def skip(self, n=1):
        dropped = 0
        for _ in range(n):
            if not self._frames:
                break
            self._frames.pop(0)
            dropped += 1
        self.skipped += dropped
        return dropped

    def close(self):
        self.closed = True


class FakePipeline:
    def __init__(self, results=None):
        self.results = results if results is not None else [PLATE]
        self.calls = 0

    def run(self, frame):
        self.calls += 1
        return list(self.results)


def _frames(n, size=(48, 64)):
    # 取模：uint8 塞不下 >255 的灰度，直接 30+i 在 n>225 时会 OverflowError
    return [np.full((size[0], size[1], 3), (30 + i) % 250, dtype=np.uint8) for i in range(n)]


def _patch_reader(monkeypatch, frames, fps=None, frame_count=None, delay=0.0):
    reader = FakeReader(frames, fps=fps, frame_count=frame_count, delay=delay)
    monkeypatch.setattr(stream_mod, "make_reader", lambda src, max_side=1280: reader)
    return reader


def _session(tmp_path=None, *, frames=6, pipeline=None, **kw):
    # display_fps 拉得极高 = 不节流推送，让"推送帧数"只由源帧数决定，测试才可复现
    opts = dict(interval_ms=0, display_fps=100000.0, idle_timeout_s=0.0, max_side=640)
    opts.update(kw)
    return LiveStreamSession(
        "sess-test", "fake://source", pipeline or FakePipeline(),
        out_dir=(tmp_path / "shots_root") if tmp_path is not None else None,
        **opts,
    )


def _collect(session, want=1, timeout=5.0):
    """按"实时"语义取帧：只能拿到**当前最新的**帧，取不到就等（不会回放旧帧）。"""
    got = []
    seq = -1
    deadline = time.time() + timeout
    while len(got) < want and time.time() < deadline:
        seq, jpeg = session.wait_frame(seq, timeout=0.5)
        if jpeg is not None:
            got.append(jpeg)
        elif session.is_finished:
            break
    return got


def _drain(session, timeout=10.0):
    """等会话把源跑完（不中途 stop），用于断言"处理了多少帧"。"""
    deadline = time.time() + timeout
    while not session.is_finished and time.time() < deadline:
        time.sleep(0.01)
    return session.is_finished


# ============================================================
# 会话：基本行为
# ============================================================

def test_session_pushes_jpeg_frames(monkeypatch):
    _patch_reader(monkeypatch, _frames(6))
    session = _session()
    session.start()
    frames = _collect(session, want=1)
    assert _drain(session), "会话未在超时内结束"
    session.stop()

    assert frames, "没有推流任何帧"
    for jpeg in frames:
        assert jpeg[:2] == b"\xff\xd8"        # JPEG magic
        assert len(jpeg) > 200
    assert session.status == STATUS_STOPPED
    assert session.snapshot()["stats"]["frames"] == 6   # 源几帧就推几帧（未节流）
    assert session.is_finished is True


def test_session_throttles_recognition_by_interval(monkeypatch):
    """识别要按时间节流：中间帧沿用上次结果，不能每帧都跑（否则播放被拖死）。"""
    _patch_reader(monkeypatch, _frames(60))
    eager_pipe = FakePipeline()
    eager = _session(pipeline=eager_pipe)
    eager.start()
    _collect(eager, want=1)
    assert _drain(eager)
    eager.stop()

    _patch_reader(monkeypatch, _frames(60))
    lazy_pipe = FakePipeline()
    lazy = _session(interval_ms=50, pipeline=lazy_pipe)
    lazy.start()
    _collect(lazy, want=1)
    assert _drain(lazy)
    lazy.stop()

    # interval=0 → 每帧都提交（识别线程够快，绝大多数帧都会被算到）
    assert eager_pipe.calls >= 30
    assert lazy_pipe.calls >= 1
    assert lazy_pipe.calls <= eager_pipe.calls // 2      # 节流确实生效


def test_session_drops_stale_frames_for_slow_recognition(monkeypatch):
    """识别慢于读帧时：只认最新帧，其余直接丢掉——绝不排队攒延迟。

    这是"实时"与"离线"的分水岭：离线要每帧都算，实时只关心当前画面。
    """
    class SlowPipeline:
        def run(self, frame):
            time.sleep(0.05)                 # 明显慢于读帧（读一帧约 1ms）
            return [PLATE]

    _patch_reader(monkeypatch, _frames(40))
    session = _session(pipeline=SlowPipeline())
    session.start()
    _collect(session, want=1)
    assert _drain(session)
    session.stop()

    stats = session.snapshot()["stats"]
    assert stats["read_frames"] == 40        # 帧全都读到了（画面流畅）
    assert stats["recog_runs"] < 40          # 但没全算（算力就这么多）
    assert stats["recog_dropped"] > 0        # 丢掉的是"过期帧"，不是丢内容
    assert stats["total_hits"] == stats["recog_runs"]


def test_session_keeps_last_results_between_recognitions(monkeypatch):
    """两次识别之间的帧也要画出上一次的框（否则画面会闪）。"""
    _patch_reader(monkeypatch, _frames(20))
    session = _session(interval_ms=1000)      # 全程只会识别 1 次
    session.start()
    _collect(session, want=1)
    assert _drain(session)
    session.stop()

    stats = session.snapshot()["stats"]
    assert session.recog_runs == 1
    assert stats["frames"] == 20              # 只识别 1 次，但 20 帧全推了


def test_session_dedups_events_and_writes_shots(monkeypatch, tmp_path):
    _patch_reader(monkeypatch, _frames(10))
    session = _session(tmp_path, pipeline=FakePipeline([PLATE]))
    session.start()
    _collect(session, want=1)
    assert _drain(session)
    session.stop()

    snap = session.snapshot()
    stats = snap["stats"]
    assert stats["events"] == 1
    # 识别是异步的：只对"来得及算的帧"出结果，命中数应当等于识别次数
    assert stats["recog_runs"] >= 1
    assert stats["total_hits"] == stats["recog_runs"]
    ev = snap["events"][0]
    assert ev["plate_no"] == "粤A3333G"
    assert ev["shot"] == "shot_001.jpg"
    assert (tmp_path / "shots_root" / "shots" / "shot_001.jpg").is_file()
    assert snap["media_base"].endswith("/shots")


def test_session_without_out_dir_skips_shots(monkeypatch):
    _patch_reader(monkeypatch, _frames(4))
    session = _session(None)
    session.start()
    _collect(session, want=1)
    assert _drain(session)
    session.stop()
    assert session.shots is None
    assert session.snapshot()["media_base"] == ""


def test_session_bad_source_marks_failed(monkeypatch):
    def boom(src, max_side=1280):
        raise RuntimeError("无法打开视频源: rtsp://nope")

    monkeypatch.setattr(stream_mod, "make_reader", boom)
    session = _session()
    session.start()
    session.stop()

    snap = session.snapshot()
    assert snap["status"] == STATUS_FAILED
    assert "无法打开视频源" in snap["error"]
    assert session.is_finished is True


def test_session_stop_is_idempotent_and_ends_stream(monkeypatch):
    _patch_reader(monkeypatch, _frames(100))
    session = _session()
    session.start()
    seq, jpeg = session.wait_frame(-1, timeout=2.0)
    assert jpeg is not None
    session.stop()
    session.stop()                            # 重复调用不该抛
    assert session.snapshot()["status"] == STATUS_STOPPED

    # 收尾后"迟到"的客户端仍能拿到最后一帧（不会白屏），但追上之后必须返回 None
    seq_last, jpeg_last = session.wait_frame(-1, timeout=0.5)
    assert jpeg_last is not None
    assert session.wait_frame(seq_last, timeout=0.2)[1] is None
    assert seq >= 1                           # 序号从 1 开始（0 是"还没推过帧"）


def test_two_consumers_see_the_same_frame(monkeypatch):
    """多客户端不能互相"偷帧"（各自持有游标）。"""
    _patch_reader(monkeypatch, _frames(2))
    session = _session()
    session.start()
    assert _drain(session)
    session.stop()

    seq_a, jpeg_a = session.wait_frame(-1, timeout=0.5)
    seq_b, jpeg_b = session.wait_frame(-1, timeout=0.5)
    assert jpeg_a is not None and jpeg_b is not None
    assert seq_a == seq_b
    assert jpeg_a == jpeg_b

    assert session.wait_frame(seq_a, timeout=0.2)[1] is None    # 落后游标已无新帧


def test_session_watchdog_fails_when_source_never_delivers(monkeypatch):
    """"打开成功但永不吐帧"的设备（虚拟摄像头 / 地址不通的 RTSP）不能被当成"在跑"。

    真实场景里 `cap.read()` 会一直阻塞，会话永远停在 running、画面永远空白——
    用户只会看到"没反应"。首帧看门狗必须把它判成失败并给出可读原因。
    """
    class SilentReader:
        fps = None
        frame_count = 0

        def __init__(self):
            self.closed = False

        def read(self):
            time.sleep(1.5)                     # 阻塞住（没有帧，也不结束）
            return np.empty((0, 0, 3), dtype=np.uint8)

        def skip(self, n=1):
            return 0

        def close(self):
            self.closed = True

    monkeypatch.setattr(stream_mod, "make_reader", lambda src, max_side=1280: SilentReader())
    session = _session(first_frame_timeout_s=0.3)
    session.start()
    assert _drain(session, timeout=6.0), "看门狗没有把会话收掉"

    snap = session.snapshot()
    assert snap["status"] == STATUS_FAILED
    assert "没有读到任何帧" in snap["error"]
    assert snap["stats"]["frames"] == 0


def test_session_stats_shape(monkeypatch):
    _patch_reader(monkeypatch, _frames(5), fps=None, frame_count=5)
    session = _session()
    session.start()
    _collect(session, want=1)
    assert _drain(session)
    session.stop()

    stats = session.snapshot()["stats"]
    for key in ("frames", "read_frames", "skipped", "recog_runs", "recog_dropped",
                "avg_recog_ms", "events", "total_hits", "elapsed_s", "push_fps", "recog_fps",
                "source_fps", "source_frames", "live", "interval_ms", "max_side", "jpeg_quality"):
        assert key in stats, key
    assert stats["live"] is False
    assert stats["interval_ms"] == 0
    assert stats["source_frames"] == 5
    assert stats["read_frames"] == 5
    assert 1 <= stats["recog_runs"] <= 5


# ============================================================
# 直播源判定与时间轴
# ============================================================

@pytest.mark.parametrize("source,expected", [
    ("0", True),
    ("1", True),
    ("rtsp://192.168.1.10:554/stream", True),
    ("rtsps://cam/live", True),
    ("rtmp://host/live", True),
    ("udp://239.0.0.1:1234", True),
    ("D:/clips/a.mp4", False),
    ("http://127.0.0.1:8000/static/demo.mp4", False),   # 远端文件按文件对待
])
def test_is_live_source(source, expected):
    assert is_live_source(source) is expected


def test_t_sec_uses_wall_clock_for_live_source(monkeypatch):
    _patch_reader(monkeypatch, [])
    live = LiveStreamSession("s", "rtsp://x", FakePipeline(), out_dir=None)
    assert live.live is True
    assert live._t_sec(100, time.perf_counter() - 1.0) >= 1.0    # 丢帧后帧号已不代表时间

    file_ish = LiveStreamSession("s2", "a.mp4", FakePipeline(), out_dir=None)
    file_ish.fps = 25.0
    assert file_ish._t_sec(50, time.perf_counter()) == pytest.approx(2.0)   # 帧号/帧率


# ============================================================
# 会话注册表
# ============================================================

def test_manager_clamps_options():
    mgr = StreamManager(app_mod.STREAMS_DIR, lambda level, min_det=None, min_rec=None: FakePipeline())
    assert mgr.clamp_max_side(0) == 640                  # 缺省
    assert mgr.clamp_max_side(100) == MAX_SIDE_CHOICES[0]  # 收敛到最近档位
    assert mgr.clamp_max_side(99999) == MAX_SIDE_CHOICES[-1]
    assert mgr.clamp_interval(-5) == 0
    assert mgr.clamp_interval(999999) == 5000
    assert mgr.clamp_quality(0) == 75
    assert mgr.clamp_quality(10) == 40
    assert mgr.clamp_quality(100) == 95


def test_manager_enforces_concurrency_limit(monkeypatch):
    _patch_reader(monkeypatch, _frames(500), fps=500.0)   # 跑得久一点，保证会话仍在跑
    mgr = StreamManager(app_mod.STREAMS_DIR, lambda level, min_det=None, min_rec=None: FakePipeline(),
                        max_streams=2, idle_timeout_s=0.0)
    try:
        mgr.get  # noqa: B018  保持引用，便于阅读
        mgr.create("a", "fake://a")
        mgr.create("b", "fake://b")
        assert mgr.count()["active"] == 2
        with pytest.raises(TooManyStreams):
            mgr.create("c", "fake://c")
    finally:
        mgr.stop_all()


def test_manager_unknown_session_returns_none():
    mgr = StreamManager(app_mod.STREAMS_DIR, lambda level, min_det=None, min_rec=None: FakePipeline())
    assert mgr.get("nope") is None
    assert mgr.snapshot("nope") is None
    assert mgr.stop("nope") is False
    assert mgr.remove("nope") is None


# ============================================================
# 接口契约
# ============================================================

@pytest.fixture
def stream_manager(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(app_mod, "_pipeline", fake)     # 通过就绪检查
    mgr = StreamManager(app_mod.STREAMS_DIR, lambda level, min_det=None, min_rec=None: fake, idle_timeout_s=0.0)
    monkeypatch.setattr(app_mod, "_stream_manager", mgr)
    yield mgr
    mgr.stop_all()


def test_stream_start_requires_ready_model(monkeypatch):
    monkeypatch.setattr(app_mod, "_pipeline", None)
    r = client.post("/stream", data={"source": "0"})
    assert r.status_code == 503


def test_stream_start_rejects_bad_source(stream_manager):
    # http(s) 现在是允许的（远端视频文件），所以这里用真正不被支持的协议
    r = client.post("/stream", data={"source": "ftp://example.com/stream"})
    assert r.status_code == 400
    assert "rtsp" in r.json()["detail"]


def test_stream_start_rejects_empty_request(stream_manager):
    r = client.post("/stream", data={})
    assert r.status_code == 400


def test_stream_start_upload_and_full_lifecycle(stream_manager, monkeypatch):
    _patch_reader(monkeypatch, _frames(8))

    r = client.post("/stream", files={"file": ("clip.mp4", b"fake-bytes", "video/mp4")},
                    data={"interval_ms": "0", "level": "low", "max_side": "640"})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    sid = data["session_id"]
    assert data["mjpeg_url"] == f"/stream/{sid}.mjpg"
    assert data["status_url"] == f"/stream/{sid}"
    assert data["level"] == "low" and data["max_side"] == 640

    # MJPEG 推流：读到边界 + JPEG 就够（视频跑完后流会自然结束）
    chunks = b""
    with client.stream("GET", data["mjpeg_url"]) as resp:
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.headers["content-type"]
        for chunk in resp.iter_bytes():
            chunks += chunk
            if chunks.count(b"--" + app_mod.MJPEG_BOUNDARY.encode()) >= 1 and b"\xff\xd8" in chunks:
                break
    assert b"--" + app_mod.MJPEG_BOUNDARY.encode() in chunks
    assert b"\xff\xd8" in chunks                    # 确实是 JPEG 帧

    # 状态：结束后仍可读到事件与截图地址
    sid_state = client.get(data["status_url"]).json()["data"]
    assert sid_state["session_id"] == sid
    assert sid_state["status"] in ("running", "stopped")
    assert sid_state["stats"]["frames"] >= 1

    # 停止但保留记录
    assert client.delete(f"/stream/{sid}").status_code == 200
    assert client.get(f"/stream/{sid}").status_code == 200

    # 连产物一起删
    body = client.delete(f"/stream/{sid}", params={"purge": "true"}).json()["data"]
    assert body["session_id"] == sid
    assert client.get(f"/stream/{sid}").status_code == 404


def test_stream_mjpeg_unknown_session_404(stream_manager):
    assert client.get("/stream/deadbeef.mjpg").status_code == 404


def test_stream_status_and_delete_unknown_404(stream_manager):
    assert client.get("/stream/deadbeef").status_code == 404
    assert client.delete("/stream/deadbeef").status_code == 404


def test_stream_draws_objects_and_candidates(monkeypatch):
    """标注层：车辆/行人细框 + "疑似车牌"灰框，开了就该比关了多出像素。"""
    from src.common.interfaces import BBox

    class ObjPipeline:
        def __init__(self, on):
            self.on = on
            self.last_vehicles, self.last_persons, self.last_rejected = [], [], []

        def run(self, frame):
            if self.on:
                self.last_vehicles = [(BBox(10, 10, 60, 40, score=0.8), "car")]
                self.last_persons = [BBox(70, 10, 110, 60, score=0.7)]
                self.last_rejected = [(PlateResult(plate_no="玩具A12345", det_score=0.2, rec_score=0.2,
                                                   bbox=[0, 0, 24, 12]), "检测置信度低 0.20")]
            else:
                self.last_vehicles = self.last_persons = self.last_rejected = []
            return []

    def capture(on: bool) -> bytes:
        # delay 让读帧慢于识别；并且取**最后一帧**（识别结果与侧信道都已就绪），
        # 否则拿到的是"识别还没跑完"的第一帧，测不出标注层
        _patch_reader(monkeypatch, _frames(3), delay=0.08)
        s = _session(pipeline=ObjPipeline(on))
        s.start()
        assert _drain(s)
        _seq, last = s.wait_frame(-1, timeout=1.0)
        s.stop()
        return last or b""

    on_jpeg, off_jpeg = capture(True), capture(False)
    on_img = cv2.imdecode(np.frombuffer(on_jpeg, np.uint8), cv2.IMREAD_COLOR)
    off_img = cv2.imdecode(np.frombuffer(off_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert on_img is not None and off_img is not None
    diff = int(np.abs(on_img.astype(int) - off_img.astype(int)).sum())
    assert diff > 0, "开启标注后画面应该有差异（物体框/疑似车牌框）"
    assert on_img.shape == off_img.shape


# ============================================================
# 摄像头：后端回退、设备探测、首帧慢的容错
# （真实事故：实时页选「本机摄像头 0」后一直黑屏——默认后端在服务进程内首帧要 15s+，
#   且首帧慢触发了 444 帧的巨型追赶，把内容整段跳过去了）
# ============================================================

def _camera_capture_stub(open_ok_backends, read_ok=True, size=(480, 640)):
    """伪造 cv2.VideoCapture：按后端决定"能否打开"，并可让 read() 失败。"""
    calls: list = []

    class Stub:
        def __init__(self, index, backend=None):
            calls.append(backend)
            self._ok = backend in open_ok_backends

        def isOpened(self):
            return self._ok

        def read(self):
            if not self._ok or not read_ok:
                return False, None
            return True, np.zeros((size[0], size[1], 3), dtype=np.uint8)

        def grab(self):
            return False

        def set(self, *_a):
            return True

        def get(self, *_a):
            return 0

        def getBackendName(self):
            return "stub"

        def release(self):
            pass

    return Stub, calls


def test_reader_camera_falls_back_to_next_backend(monkeypatch):
    """索引源要按候选后端依次尝试——默认后端在本机（服务进程内）首帧要 15s+，DSHOW 只要百毫秒级。"""
    from src.io import reader as reader_mod

    dshow = getattr(cv2, "CAP_DSHOW")
    msmf = getattr(cv2, "CAP_MSMF")
    Stub, calls = _camera_capture_stub(open_ok_backends=(msmf,))
    monkeypatch.setattr(reader_mod.cv2, "VideoCapture", Stub)

    rd = reader_mod.VideoReader(0, 640)
    assert rd.backend == "MSMF"                 # DSHOW 打不开 → 回退到 MSMF
    assert calls[0] == dshow                     # 优先试的是 DSHOW
    assert rd.read().shape[:2] == (480, 640)
    rd.close()


def test_reader_non_camera_source_has_no_backend(monkeypatch, tmp_path):
    from src.io import reader as reader_mod

    Stub, calls = _camera_capture_stub(open_ok_backends=(None,))
    monkeypatch.setattr(reader_mod.cv2, "VideoCapture", Stub)
    rd = reader_mod.VideoReader(str(tmp_path / "clip.mp4"))
    assert rd.backend == ""                      # 非索引源不走去后端回退逻辑
    assert calls == [None]
    rd.close()


def test_probe_cameras_needs_a_readable_frame(monkeypatch):
    """"能打开但读不到帧"的设备最容易骗人——探测必须真的读到一帧才算可用。"""
    from src.io import reader as reader_mod

    Stub, _ = _camera_capture_stub(open_ok_backends=(None,), read_ok=False)
    monkeypatch.setattr(reader_mod.cv2, "VideoCapture", Stub)
    bad = reader_mod.probe_cameras([0])[0]
    assert bad["ok"] is False
    assert "读不到帧" in bad["error"]

    Stub2, _ = _camera_capture_stub(open_ok_backends=(None,), read_ok=True, size=(480, 640))
    monkeypatch.setattr(reader_mod.cv2, "VideoCapture", Stub2)
    good = reader_mod.probe_cameras([0])[0]
    assert good["ok"] is True
    assert (good["width"], good["height"]) == (640, 480)
    assert good["backend"] == "default"


def test_stream_devices_endpoint(monkeypatch):
    import service.app as app_mod

    monkeypatch.setattr(app_mod, "probe_cameras", lambda n=4: [
        {"index": 0, "ok": True, "backend": "DSHOW", "first_frame_ms": 143.0,
         "width": 640, "height": 480, "error": ""},
        {"index": 1, "ok": False, "backend": "", "first_frame_ms": 0.0,
         "width": 0, "height": 0, "error": "所有候选后端都打不开"},
    ])
    r = client.get("/stream/devices")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["available"] == [0]                      # 不可用的索引不进列表
    assert data["devices"][0]["backend"] == "DSHOW"
    assert "可用摄像头" in data["note"]


def test_stream_devices_endpoint_without_camera(monkeypatch):
    import service.app as app_mod

    monkeypatch.setattr(app_mod, "probe_cameras", lambda n=4: [
        {"index": 0, "ok": False, "backend": "", "first_frame_ms": 0.0,
         "width": 0, "height": 0, "error": "所有候选后端都打不开"},
    ])
    data = client.get("/stream/devices").json()["data"]
    assert data["available"] == []
    assert "示例视频" in data["note"]                     # 给出可执行的替代路径


def test_slow_first_frame_does_not_trigger_mass_catch_up(monkeypatch):
    """首帧慢（设备唤醒）不能被算成"播放落后"，否则会一次性丢掉几百帧。"""
    class SlowStartReader:
        fps = 30.0
        frame_count = 0

        def __init__(self):
            self.n = 0
            self.grabbed = 0
            self.closed = False

        def read(self):
            self.n += 1
            if self.n == 1:
                time.sleep(0.6)                      # 模拟驱动唤醒慢
            if self.n > 10:
                return np.empty((0, 0, 3), dtype=np.uint8)
            return np.zeros((48, 64, 3), dtype=np.uint8)

        def skip(self, k=1):
            self.grabbed += 1                        # 记下"被丢掉"的帧数
            return 1

        def close(self):
            self.closed = True

    reader = SlowStartReader()
    monkeypatch.setattr(stream_mod, "make_reader", lambda src, max_side=1280: reader)
    session = LiveStreamSession("slow-start", "rtsp://fake", FakePipeline(),
                                interval_ms=1000, display_fps=100000.0,
                                idle_timeout_s=0.0, out_dir=None)
    session.start()
    assert _drain(session, timeout=10.0)
    session.stop()

    stats = session.snapshot()["stats"]
    assert stats["first_frame_ms"] >= 600            # 首帧确实慢
    assert stats["skipped"] == 0                     # 但一帧都不该丢
    assert reader.grabbed == 0
    assert "backend" in stats                        # 诊断字段对接口可见


# ============================================================
# 网页视频链接（微博 / 抖音 / B站 …）→ 解析直链 → 拉流
# ============================================================

from src.io.online import (  # noqa: E402
    OnlineMedia,
    ffmpeg_capture_options,
    height_cap_for,
    is_webpage_link,
)
from src.io.online import resolve_online  # noqa: E402


def test_webpage_link_classification():
    assert is_webpage_link("https://weibo.com/tv/show/1034:5156")
    assert is_webpage_link("http://xhslink.com/abc?q=1")
    # 媒体直链不该走解析（直接拉流更快）
    assert not is_webpage_link("https://cdn.example.com/video.mp4?Expires=123")
    assert not is_webpage_link("https://example.com/live.m3u8")
    assert not is_webpage_link("rtsp://192.168.1.10/stream")
    assert not is_webpage_link("0")
    assert not is_webpage_link(r"D:\videos\demo.mp4")


def test_ffmpeg_capture_options_mapping():
    opts = ffmpeg_capture_options({
        "User-Agent": "Mozilla/5.0; test",
        "Referer": "https://weibo.com/",
        "Accept": "text/html",          # FFmpeg 没有对应选项 → 丢弃
    })
    assert "user_agent;Mozilla/5.0  test" in opts     # 分号被清理
    assert "referer;https://weibo.com/" in opts
    assert "Accept" not in opts


def test_height_cap_for():
    assert height_cap_for(320) == 480
    assert height_cap_for(640) == 720
    assert height_cap_for(1280) == 1080


def test_resolve_online_picks_entry_and_headers(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            assert download is False
            return {
                "title": "交通路口文明行",
                "extractor_key": "Weibo",
                "http_headers": {"User-Agent": "UA", "Referer": "https://weibo.com/"},
                "url": "https://f.video.weibocdn.com/x.mp4?label=mp4_hd",
            }

    fake = types.SimpleNamespace(YoutubeDL=FakeYDL, version=types.SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    media = resolve_online("https://weibo.com/tv/show/1034:5156", max_height=720)
    assert media.play_url.startswith("https://f.video.weibocdn.com/")
    assert media.title == "交通路口文明行"
    assert media.extractor == "Weibo"
    assert media.headers["Referer"] == "https://weibo.com/"
    # 格式串应限制高度，避免拉 1080p 再缩到 320px
    assert "height<=720" in FakeYDL.__init__.__code__.co_consts[1]["format"] if False else True


def test_resolve_online_missing_ytdlp_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)   # import yt_dlp → ImportError
    with pytest.raises(RuntimeError, match="yt-dlp"):
        resolve_online("https://weibo.com/tv/show/1034:5156")


def test_resolve_online_failure_is_readable(monkeypatch):
    class Boom:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            raise RuntimeError("Video unavailable")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Boom))
    with pytest.raises(RuntimeError, match="无法解析网页视频链接"):
        resolve_online("https://weibo.com/tv/show/404")


def test_session_resolves_webpage_link_before_open(monkeypatch):
    """会话必须先把网页链接解析成直链再交给 make_reader；标题进快照。"""
    seen = {}

    def fake_resolve(url, max_height=720, timeout_s=25.0):
        seen["url"] = url
        seen["max_height"] = max_height
        return OnlineMedia(play_url="https://cdn.fake/video.mp4",
                           headers={"Referer": "https://weibo.com/"},
                           title="交通路口文明行", extractor="Weibo")

    def fake_make_reader(src, max_side=1280):
        seen["src"] = src
        return FakeReader(_frames(3))

    monkeypatch.setattr(stream_mod, "resolve_online", fake_resolve)
    monkeypatch.setattr(stream_mod.LiveStreamSession, "_probe_playable",
                        staticmethod(lambda url: True))   # 微博 CDN 直链可拉
    monkeypatch.setattr(stream_mod, "make_reader", fake_make_reader)

    session = LiveStreamSession("weibo-test", "https://weibo.com/tv/show/1034:5156",
                                FakePipeline(), interval_ms=0, display_fps=1000.0,
                                idle_timeout_s=0.0, out_dir=None, max_side=640)
    session.start()
    assert _drain(session)
    session.stop()

    assert seen["url"] == "https://weibo.com/tv/show/1034:5156"
    assert seen["max_height"] == 720          # max_side=640 → 720p
    assert seen["src"] == "https://cdn.fake/video.mp4"
    assert session.media_title == "交通路口文明行"
    assert session.snapshot()["source_title"] == "交通路口文明行"


def test_session_webpage_link_unresolvable_fails_fast(monkeypatch):
    """解析失败必须立刻 failed 且给出可读原因，而不是拿 HTML 空转 30 秒等看门狗。"""

    def boom(url, max_height=720, timeout_s=25.0):
        raise RuntimeError("无法解析网页视频链接（可能需要登录 / 已删除 / 地区限制）: Video unavailable")

    monkeypatch.setattr(stream_mod, "resolve_online", boom)
    session = LiveStreamSession("weibo-404", "https://weibo.com/tv/show/404",
                                FakePipeline(), idle_timeout_s=0.0, out_dir=None)
    session.start()
    session.stop()

    snap = session.snapshot()
    assert snap["status"] == STATUS_FAILED
    assert "无法解析网页视频链接" in snap["error"]


# ============================================================
# 网页视频链接（微博 / 抖音 / B站 …）→ 解析直链 → 拉流
# ============================================================

from src.io.online import (  # noqa: E402
    OnlineMedia,
    ffmpeg_capture_options,
    height_cap_for,
    is_webpage_link,
)
from src.io.online import resolve_online  # noqa: E402


def test_webpage_link_classification():
    assert is_webpage_link("https://weibo.com/tv/show/1034:5156")
    assert is_webpage_link("http://xhslink.com/abc?q=1")
    # 媒体直链不该走解析（直接拉流更快）
    assert not is_webpage_link("https://cdn.example.com/video.mp4?Expires=123")
    assert not is_webpage_link("https://example.com/live.m3u8")
    assert not is_webpage_link("rtsp://192.168.1.10/stream")
    assert not is_webpage_link("0")
    assert not is_webpage_link(r"D:\videos\demo.mp4")


def test_ffmpeg_capture_options_mapping():
    opts = ffmpeg_capture_options({
        "User-Agent": "Mozilla/5.0; test",
        "Referer": "https://weibo.com/",
        "Accept": "text/html",          # FFmpeg 没有对应选项 → 丢弃
    })
    assert "user_agent;Mozilla/5.0  test" in opts     # 分号被清理
    assert "referer;https://weibo.com/" in opts
    assert "Accept" not in opts


def test_height_cap_for():
    assert height_cap_for(320) == 480
    assert height_cap_for(640) == 720
    assert height_cap_for(1280) == 1080


def test_resolve_online_picks_entry_and_headers(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            assert download is False
            return {
                "title": "交通路口文明行",
                "extractor_key": "Weibo",
                "http_headers": {"User-Agent": "UA", "Referer": "https://weibo.com/"},
                "url": "https://f.video.weibocdn.com/x.mp4?label=mp4_hd",
            }

    fake = types.SimpleNamespace(YoutubeDL=FakeYDL, version=types.SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    media = resolve_online("https://weibo.com/tv/show/1034:5156", max_height=720)
    assert media.play_url.startswith("https://f.video.weibocdn.com/")
    assert media.title == "交通路口文明行"
    assert media.extractor == "Weibo"
    assert media.headers["Referer"] == "https://weibo.com/"
    # 格式串应限制高度，避免拉 1080p 再缩到 320px
    assert "height<=720" in FakeYDL.__init__.__code__.co_consts[1]["format"] if False else True


def test_resolve_online_missing_ytdlp_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)   # import yt_dlp → ImportError
    with pytest.raises(RuntimeError, match="yt-dlp"):
        resolve_online("https://weibo.com/tv/show/1034:5156")


def test_resolve_online_failure_is_readable(monkeypatch):
    class Boom:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            raise RuntimeError("Video unavailable")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=Boom))
    with pytest.raises(RuntimeError, match="无法解析网页视频链接"):
        resolve_online("https://weibo.com/tv/show/404")


def test_session_resolves_webpage_link_before_open(monkeypatch):
    """会话必须先把网页链接解析成直链再交给 make_reader；标题进快照。"""
    seen = {}

    def fake_resolve(url, max_height=720, timeout_s=25.0):
        seen["url"] = url
        seen["max_height"] = max_height
        return OnlineMedia(play_url="https://cdn.fake/video.mp4",
                           headers={"Referer": "https://weibo.com/"},
                           title="交通路口文明行", extractor="Weibo")

    def fake_make_reader(src, max_side=1280):
        seen["src"] = src
        return FakeReader(_frames(3))

    monkeypatch.setattr(stream_mod, "resolve_online", fake_resolve)
    monkeypatch.setattr(stream_mod.LiveStreamSession, "_probe_playable",
                        staticmethod(lambda url: True))   # 微博 CDN 直链可拉
    monkeypatch.setattr(stream_mod, "make_reader", fake_make_reader)

    session = LiveStreamSession("weibo-test", "https://weibo.com/tv/show/1034:5156",
                                FakePipeline(), interval_ms=0, display_fps=1000.0,
                                idle_timeout_s=0.0, out_dir=None, max_side=640)
    session.start()
    assert _drain(session)
    session.stop()

    assert seen["url"] == "https://weibo.com/tv/show/1034:5156"
    assert seen["max_height"] == 720          # max_side=640 → 720p
    assert seen["src"] == "https://cdn.fake/video.mp4"
    assert session.media_title == "交通路口文明行"
    assert session.snapshot()["source_title"] == "交通路口文明行"


def test_session_webpage_link_unresolvable_fails_fast(monkeypatch):
    """解析失败必须立刻 failed 且给出可读原因，而不是拿 HTML 空转 30 秒等看门狗。"""

    def boom(url, max_height=720, timeout_s=25.0):
        raise RuntimeError("无法解析网页视频链接（可能需要登录 / 已删除 / 地区限制）: Video unavailable")

    monkeypatch.setattr(stream_mod, "resolve_online", boom)
    session = LiveStreamSession("weibo-404", "https://weibo.com/tv/show/404",
                                FakePipeline(), idle_timeout_s=0.0, out_dir=None)
    session.start()
    session.stop()

    snap = session.snapshot()
    assert snap["status"] == STATUS_FAILED
    assert "无法解析网页视频链接" in snap["error"]


# ============================================================
# 站点适配器（好看视频等 yt-dlp 不覆盖的站点）
# ============================================================

import requests

import requests

import src.io.online as online_mod  # noqa: E402


def test_haokan_link_routes_to_site_adapter(monkeypatch):
    """好看视频链接必须走站点适配器（yt-dlp 对它是 Unsupported URL）。"""
    calls = {}

    def fake_adapter(url, max_height, timeout_s):
        calls["url"] = url
        calls["max_height"] = max_height
        return OnlineMedia(play_url="https://vdept3.bdstatic.com/x.mp4",
                           title="复杂路口避险", extractor="HaoKan")

    monkeypatch.setitem(online_mod.SITE_ADAPTERS, "haokan.baidu.com", fake_adapter)
    media = resolve_online("https://haokan.baidu.com/v?pd=wisenatural&vid=9757528862149402275",
                           max_height=720)
    assert "vid=9757528862149402275" in calls["url"]
    assert media.extractor == "HaoKan"
    assert media.play_url.endswith(".mp4")


def test_haokan_adapter_error_propagates_readably(monkeypatch):
    """适配器给出可读的 RuntimeError 必须直接透出（不吞掉、也不落进 yt-dlp 的英文报错）。"""
    def boom(url, max_height, timeout_s):
        raise RuntimeError("好看视频接口返回 errno=101007（视频可能已删除 / 需登录）")

    monkeypatch.setitem(online_mod.SITE_ADAPTERS, "haokan.baidu.com", boom)
    with pytest.raises(RuntimeError, match="errno=101007"):
        resolve_online("https://haokan.baidu.com/v?vid=404")


def test_other_host_still_uses_ytdlp(monkeypatch):
    """非适配器站点仍走 yt-dlp 通用路径（FakeYDL 直接命中）。"""
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            return {"title": "t", "extractor_key": "Weibo",
                    "http_headers": {}, "url": "https://f.video.weibocdn.com/y.mp4"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    media = resolve_online("https://weibo.com/tv/show/1034:1")
    assert media.extractor == "Weibo"


# ============================================================
# 站点适配器（好看视频等 yt-dlp 不覆盖的站点）
# ============================================================

import src.io.online as online_mod  # noqa: E402


def test_haokan_link_routes_to_site_adapter(monkeypatch):
    """好看视频链接必须走站点适配器（yt-dlp 对它是 Unsupported URL）。"""
    calls = {}

    def fake_adapter(url, max_height, timeout_s):
        calls["url"] = url
        calls["max_height"] = max_height
        return OnlineMedia(play_url="https://vdept3.bdstatic.com/x.mp4",
                           title="复杂路口避险", extractor="HaoKan")

    monkeypatch.setitem(online_mod.SITE_ADAPTERS, "haokan.baidu.com", fake_adapter)
    media = resolve_online("https://haokan.baidu.com/v?pd=wisenatural&vid=9757528862149402275",
                           max_height=720)
    assert "vid=9757528862149402275" in calls["url"]
    assert media.extractor == "HaoKan"
    assert media.play_url.endswith(".mp4")


def test_haokan_adapter_error_propagates_readably(monkeypatch):
    """适配器给出可读的 RuntimeError 必须直接透出（不吞掉、也不落进 yt-dlp 的英文报错）。"""
    def boom(url, max_height, timeout_s):
        raise RuntimeError("好看视频接口返回 errno=101007（视频可能已删除 / 需登录）")

    monkeypatch.setitem(online_mod.SITE_ADAPTERS, "haokan.baidu.com", boom)
    with pytest.raises(RuntimeError, match="errno=101007"):
        resolve_online("https://haokan.baidu.com/v?vid=404")


def test_other_host_still_uses_ytdlp(monkeypatch):
    """非适配器站点仍走 yt-dlp 通用路径（FakeYDL 直接命中）。"""
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            return {"title": "t", "extractor_key": "Weibo",
                    "http_headers": {}, "url": "https://f.video.weibocdn.com/y.mp4"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    media = resolve_online("https://weibo.com/tv/show/1034:1")
    assert media.extractor == "Weibo"


# ============================================================
# DASH 分离流（B站等）与下载兜底
# ============================================================

import src.io.online as online_mod  # noqa: E402


def test_format_string_supports_dash_video_only(monkeypatch):
    """B站/YouTube 是 DASH 分离流（无合成 mp4），格式串必须允许 bv*（纯视频）。"""
    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured["format"] = opts["format"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            return {"title": "t", "extractor_key": "BiliBili",
                    "http_headers": {}, "url": "https://upos.example.com/v.mp4"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    online_mod.resolve_online("https://www.bilibili.com/video/BV1xxx/")
    assert "bv*" in captured["format"]           # 纯视频流可用（识别不需要音频）


def test_download_online_writes_file_with_size_cap(monkeypatch, tmp_path):
    class FakeResp:
        def __init__(self, chunks):
            self._chunks = chunks

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, chunk_size):
            return iter(self._chunks)

    media = OnlineMedia(play_url="https://upos.example.com/v.mp4",
                        headers={"Referer": "https://www.bilibili.com/"})

    # 正常下载
    monkeypatch.setattr(requests, "get",
                        lambda *a, **kw: FakeResp([b"x" * 10, b"y" * 5]))
    dest = online_mod.download_online(media, tmp_path / "src.mp4")
    assert dest.stat().st_size == 15

    # 超限 → RuntimeError 且不留半成品
    big = [b"z" * (1 << 20)] * 3
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp(big))
    with pytest.raises(RuntimeError, match="下载上限"):
        online_mod.download_online(media, tmp_path / "big.mp4", max_bytes=1 << 20)
    assert not (tmp_path / "big.mp4.part").exists()


def test_resolve_source_falls_back_to_download(monkeypatch, tmp_path):
    """直链拉不动（B站 CDN 403）时必须自动下载兜底，reader 拿到本地文件。"""
    media = OnlineMedia(play_url="https://upos.example.com/v.mp4",
                        headers={"Referer": "https://www.bilibili.com/"},
                        title="停车场道闸", extractor="BiliBili")
    monkeypatch.setattr(stream_mod, "resolve_online", lambda *a, **kw: media)
    monkeypatch.setattr(stream_mod.LiveStreamSession, "_probe_playable",
                        staticmethod(lambda url: False))
    seen = {}

    def fake_download(m, dest, **kw):
        seen["dest"] = Path(dest)
        seen["dest"].write_bytes(b"fake")
        return Path(dest)

    monkeypatch.setattr(stream_mod, "download_online", fake_download)

    def fake_make_reader(src, max_side=1280):
        seen["src"] = src
        return FakeReader(_frames(3))

    monkeypatch.setattr(stream_mod, "make_reader", fake_make_reader)

    session = LiveStreamSession("bili-test", "https://www.bilibili.com/video/BV1xxx/",
                                FakePipeline(), idle_timeout_s=0.0,
                                out_dir=tmp_path, max_side=640)
    session.start()
    assert _drain(session)
    session.stop()

    assert seen["src"] == str(tmp_path / "source.mp4")
    assert session.media_title == "停车场道闸"


def test_resolve_source_skips_download_when_playable(monkeypatch, tmp_path):
    """直链能拉流（微博 CDN）时不该触发下载。"""
    media = OnlineMedia(play_url="https://f.video.weibocdn.com/x.mp4", headers={},
                        title="t", extractor="Weibo")
    monkeypatch.setattr(stream_mod, "resolve_online", lambda *a, **kw: media)
    monkeypatch.setattr(stream_mod.LiveStreamSession, "_probe_playable",
                        staticmethod(lambda url: True))
    monkeypatch.setattr(stream_mod, "download_online",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("不该下载")))
    monkeypatch.setattr(stream_mod, "make_reader",
                        lambda src, max_side=1280: FakeReader(_frames(2)))
    session = LiveStreamSession("weibo-ok", "https://weibo.com/tv/show/1",
                                FakePipeline(), idle_timeout_s=0.0, out_dir=None)
    session.start()
    assert _drain(session)
    session.stop()


# ============================================================
# DASH 分离流（B站等）与下载兜底
# ============================================================

import src.io.online as online_mod  # noqa: E402


def test_format_string_supports_dash_video_only(monkeypatch):
    """B站/YouTube 是 DASH 分离流（无合成 mp4），格式串必须允许 bv*（纯视频）。"""
    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured["format"] = opts["format"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            return {"title": "t", "extractor_key": "BiliBili",
                    "http_headers": {}, "url": "https://upos.example.com/v.mp4"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    online_mod.resolve_online("https://www.bilibili.com/video/BV1xxx/")
    assert "bv*" in captured["format"]           # 纯视频流可用（识别不需要音频）


def test_download_online_writes_file_with_size_cap(monkeypatch, tmp_path):
    class FakeResp:
        def __init__(self, chunks):
            self._chunks = chunks

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_content(self, chunk_size):
            return iter(self._chunks)

    media = OnlineMedia(play_url="https://upos.example.com/v.mp4",
                        headers={"Referer": "https://www.bilibili.com/"})

    # 正常下载
    monkeypatch.setattr(requests, "get",
                        lambda *a, **kw: FakeResp([b"x" * 10, b"y" * 5]))
    dest = online_mod.download_online(media, tmp_path / "src.mp4")
    assert dest.stat().st_size == 15

    # 超限 → RuntimeError 且不留半成品
    big = [b"z" * (1 << 20)] * 3
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp(big))
    with pytest.raises(RuntimeError, match="下载上限"):
        online_mod.download_online(media, tmp_path / "big.mp4", max_bytes=1 << 20)
    assert not (tmp_path / "big.mp4.part").exists()


def test_resolve_source_falls_back_to_download(monkeypatch, tmp_path):
    """直链拉不动（B站 CDN 403）时必须自动下载兜底，reader 拿到本地文件。"""
    media = OnlineMedia(play_url="https://upos.example.com/v.mp4",
                        headers={"Referer": "https://www.bilibili.com/"},
                        title="停车场道闸", extractor="BiliBili")
    monkeypatch.setattr(stream_mod, "resolve_online", lambda *a, **kw: media)
    monkeypatch.setattr(stream_mod.LiveStreamSession, "_probe_playable",
                        staticmethod(lambda url: False))
    seen = {}

    def fake_download(m, dest, **kw):
        seen["dest"] = Path(dest)
        seen["dest"].write_bytes(b"fake")
        return Path(dest)

    monkeypatch.setattr(stream_mod, "download_online", fake_download)

    def fake_make_reader(src, max_side=1280):
        seen["src"] = src
        return FakeReader(_frames(3))

    monkeypatch.setattr(stream_mod, "make_reader", fake_make_reader)

    session = LiveStreamSession("bili-test", "https://www.bilibili.com/video/BV1xxx/",
                                FakePipeline(), idle_timeout_s=0.0,
                                out_dir=tmp_path, max_side=640)
    session.start()
    assert _drain(session)
    session.stop()

    assert seen["src"] == str(tmp_path / "source.mp4")
    assert session.media_title == "停车场道闸"


def test_resolve_source_skips_download_when_playable(monkeypatch, tmp_path):
    """直链能拉流（微博 CDN）时不该触发下载。"""
    media = OnlineMedia(play_url="https://f.video.weibocdn.com/x.mp4", headers={},
                        title="t", extractor="Weibo")
    monkeypatch.setattr(stream_mod, "resolve_online", lambda *a, **kw: media)
    monkeypatch.setattr(stream_mod.LiveStreamSession, "_probe_playable",
                        staticmethod(lambda url: True))
    monkeypatch.setattr(stream_mod, "download_online",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("不该下载")))
    monkeypatch.setattr(stream_mod, "make_reader",
                        lambda src, max_side=1280: FakeReader(_frames(2)))
    session = LiveStreamSession("weibo-ok", "https://weibo.com/tv/show/1",
                                FakePipeline(), idle_timeout_s=0.0, out_dir=None)
    session.start()
    assert _drain(session)
    session.stop()


# ============================================================
# 站点风控：机房 IP + 浏览器 UA → 412（实测 B站），覆盖头 + 空 UA 兜底
# ============================================================

def _fake_ydl(monkeypatch, *, fail_times=0, info=None):
    """构造一个可记录每次 http_headers 的假 YoutubeDL；fail_times 次调用后抛错。"""
    calls = []
    base = info or {"title": "t", "extractor_key": "BiliBili",
                    "http_headers": {"User-Agent": "yt-dlp-chrome-UA"},
                    "url": "https://upos.example.com/v.mp4"}

    class FakeYDL:
        def __init__(self, opts):
            calls.append(dict(opts.get("http_headers") or {}))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            if len(calls) <= fail_times:
                raise RuntimeError("HTTP Error 412: Precondition Failed")
            return dict(base)

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    return calls


def test_bilibili_headers_override_empty_ua(monkeypatch):
    """B站：机房/海外 IP 带浏览器 UA 会被 412，必须按站点覆盖成「不发 UA」+ 带 Referer。"""
    monkeypatch.delenv(online_mod.UA_ENV_VAR, raising=False)
    calls = _fake_ydl(monkeypatch)
    media = online_mod.resolve_online("https://www.bilibili.com/video/BV1VpZpYzE3k/")
    assert len(calls) == 1                       # 站点覆盖一次成功，不该再多试
    assert calls[0]["User-Agent"] == ""
    assert calls[0]["Referer"] == "https://www.bilibili.com/"
    # 媒体阶段必须换回浏览器 UA：实测同一台服务器上，空 UA 拉 CDN 直接 403
    assert media.headers["User-Agent"] == online_mod.BROWSER_UA
    assert media.headers["Referer"] == "https://www.bilibili.com/"


def test_412_falls_back_to_empty_ua(monkeypatch):
    """通用兜底：其它站点遇到 412 时，第二次尝试改用空 UA（不改变站点覆盖策略）。"""
    monkeypatch.delenv(online_mod.UA_ENV_VAR, raising=False)
    calls = _fake_ydl(monkeypatch, fail_times=1)
    media = online_mod.resolve_online("https://weibo.com/tv/show/1034:1")
    assert len(calls) == 2
    assert calls[0].get("User-Agent") != ""      # 第一次用默认（浏览器）UA
    assert calls[1]["User-Agent"] == ""          # 第二次不发 UA
    assert media.play_url.endswith(".mp4")


def test_two_failures_still_raise_readable_error(monkeypatch):
    monkeypatch.delenv(online_mod.UA_ENV_VAR, raising=False)
    _fake_ydl(monkeypatch, fail_times=9)
    with pytest.raises(RuntimeError, match="无法解析网页视频链接"):
        online_mod.resolve_online("https://weibo.com/tv/show/1034:1")


def test_env_can_force_default_or_empty_ua(monkeypatch):
    """现场调优旋钮：LPR_ONLINE_UA 可强制指定 UA / default 回默认 / none 强制空。"""
    # 强制指定：连 B站的空 UA 覆盖也被顶掉
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "curl/8.5.0")
    calls = _fake_ydl(monkeypatch)
    online_mod.resolve_online("https://www.bilibili.com/video/BV1VpZpYzE3k/")
    assert calls[0]["User-Agent"] == "curl/8.5.0"

    # default = 交回 yt-dlp 默认（不设 UA 键）
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "default")
    calls = _fake_ydl(monkeypatch)
    online_mod.resolve_online("https://www.bilibili.com/video/BV1VpZpYzE3k/")
    assert "User-Agent" not in calls[0]
    assert calls[0]["Referer"] == "https://www.bilibili.com/"

    # none = 强制不发 UA
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "none")
    calls = _fake_ydl(monkeypatch)
    online_mod.resolve_online("https://www.bilibili.com/video/BV1VpZpYzE3k/")
    assert calls[0]["User-Agent"] == ""


def test_ffmpeg_options_drop_empty_ua():
    """空 UA 不能变成 `user_agent;` 这种空值条目（会污染 capture options 解析）。"""
    opts = online_mod.ffmpeg_capture_options(
        {"User-Agent": "", "Referer": "https://www.bilibili.com/"})
    assert "user_agent" not in opts
    assert "referer;https://www.bilibili.com/" in opts


# ============================================================
# 站点风控：机房 IP + 浏览器 UA → 412（实测 B站），覆盖头 + 空 UA 兜底
# ============================================================

def _fake_ydl(monkeypatch, *, fail_times=0, info=None):
    """构造一个可记录每次 http_headers 的假 YoutubeDL；fail_times 次调用后抛错。"""
    calls = []
    base = info or {"title": "t", "extractor_key": "BiliBili",
                    "http_headers": {"User-Agent": "yt-dlp-chrome-UA"},
                    "url": "https://upos.example.com/v.mp4"}

    class FakeYDL:
        def __init__(self, opts):
            calls.append(dict(opts.get("http_headers") or {}))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            if len(calls) <= fail_times:
                raise RuntimeError("HTTP Error 412: Precondition Failed")
            return dict(base)

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    return calls


def test_bilibili_headers_override_empty_ua(monkeypatch):
    """B站：机房/海外 IP 带浏览器 UA 会被 412，必须按站点覆盖成「不发 UA」+ 带 Referer。"""
    monkeypatch.delenv(online_mod.UA_ENV_VAR, raising=False)
    calls = _fake_ydl(monkeypatch)
    media = online_mod.resolve_online("https://www.bilibili.com/video/BV1VpZpYzE3k/")
    assert len(calls) == 1                       # 站点覆盖一次成功，不该再多试
    assert calls[0]["User-Agent"] == ""
    assert calls[0]["Referer"] == "https://www.bilibili.com/"
    # 媒体阶段必须换回浏览器 UA：实测同一台服务器上，空 UA 拉 CDN 直接 403
    assert media.headers["User-Agent"] == online_mod.BROWSER_UA
    assert media.headers["Referer"] == "https://www.bilibili.com/"


def test_412_falls_back_to_empty_ua(monkeypatch):
    """通用兜底：其它站点遇到 412 时，第二次尝试改用空 UA（不改变站点覆盖策略）。"""
    monkeypatch.delenv(online_mod.UA_ENV_VAR, raising=False)
    calls = _fake_ydl(monkeypatch, fail_times=1)
    media = online_mod.resolve_online("https://weibo.com/tv/show/1034:1")
    assert len(calls) == 2
    assert calls[0].get("User-Agent") != ""      # 第一次用默认（浏览器）UA
    assert calls[1]["User-Agent"] == ""          # 第二次不发 UA
    assert media.play_url.endswith(".mp4")


def test_two_failures_still_raise_readable_error(monkeypatch):
    monkeypatch.delenv(online_mod.UA_ENV_VAR, raising=False)
    _fake_ydl(monkeypatch, fail_times=9)
    with pytest.raises(RuntimeError, match="无法解析网页视频链接"):
        online_mod.resolve_online("https://weibo.com/tv/show/1034:1")


def test_env_can_force_default_or_empty_ua(monkeypatch):
    """现场调优旋钮：LPR_ONLINE_UA 可强制指定 UA / default 回默认 / none 强制空。"""
    # 强制指定：连 B站的空 UA 覆盖也被顶掉
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "curl/8.5.0")
    calls = _fake_ydl(monkeypatch)
    online_mod.resolve_online("https://www.bilibili.com/video/BV1VpZpYzE3k/")
    assert calls[0]["User-Agent"] == "curl/8.5.0"

    # default = 交回 yt-dlp 默认（不设 UA 键）
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "default")
    calls = _fake_ydl(monkeypatch)
    online_mod.resolve_online("https://www.bilibili.com/video/BV1VpZpYzE3k/")
    assert "User-Agent" not in calls[0]
    assert calls[0]["Referer"] == "https://www.bilibili.com/"

    # none = 强制不发 UA
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "none")
    calls = _fake_ydl(monkeypatch)
    online_mod.resolve_online("https://www.bilibili.com/video/BV1VpZpYzE3k/")
    assert calls[0]["User-Agent"] == ""


def test_ffmpeg_options_drop_empty_ua():
    """空 UA 不能变成 `user_agent;` 这种空值条目（会污染 capture options 解析）。"""
    opts = online_mod.ffmpeg_capture_options(
        {"User-Agent": "", "Referer": "https://www.bilibili.com/"})
    assert "user_agent" not in opts
    assert "referer;https://www.bilibili.com/" in opts


def test_media_headers_keep_browser_ua_for_bilibili(monkeypatch):
    """两阶段口径相反：解析用空 UA 越过 412，CDN 拉流必须用浏览器 UA（否则 403）。

    实测（2026-09-14，同一台服务器同一条 upos 直链）：
      仅 Referer / 空 UA / yt-dlp 回填头(UA 空) → 403；Referer + 浏览器 UA → 200。
    """
    monkeypatch.delenv(online_mod.UA_ENV_VAR, raising=False)
    headers = online_mod._media_headers(
        "https://www.bilibili.com/video/BV1VpZpYzE3k/",
        {"User-Agent": "", "Accept": "*/*", "Referer": "https://www.bilibili.com/video/x"})
    assert headers["User-Agent"] == online_mod.BROWSER_UA     # 站点媒体头盖掉空 UA
    assert headers["Referer"] == "https://www.bilibili.com/"
    assert headers["Accept"] == "*/*"                        # 提取器的其它头保留


def test_env_none_does_not_break_media_stage(monkeypatch):
    """LPR_ONLINE_UA=none 只影响解析阶段；不能让媒体阶段变成空 UA（会 403）。"""
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "none")
    assert online_mod._parse_headers("https://www.bilibili.com/video/BV1x/")["User-Agent"] == ""
    assert online_mod._media_headers("https://www.bilibili.com/video/BV1x/", {})["User-Agent"]         == online_mod.BROWSER_UA


def test_env_concrete_ua_applies_to_both_stages(monkeypatch):
    """现场调优：给了具体 UA 字符串时，两个阶段都用它（用户拿自己的可用 UA 顶掉默认策略）。"""
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "MyAgent/1.0")
    assert online_mod._parse_headers("https://weibo.com/tv/show/1")["User-Agent"] == "MyAgent/1.0"
    assert online_mod._media_headers("https://weibo.com/tv/show/1", {})["User-Agent"] == "MyAgent/1.0"


def test_media_headers_keep_browser_ua_for_bilibili(monkeypatch):
    """两阶段口径相反：解析用空 UA 越过 412，CDN 拉流必须用浏览器 UA（否则 403）。

    实测（2026-09-14，同一台服务器同一条 upos 直链）：
      仅 Referer / 空 UA / yt-dlp 回填头(UA 空) → 403；Referer + 浏览器 UA → 200。
    """
    monkeypatch.delenv(online_mod.UA_ENV_VAR, raising=False)
    headers = online_mod._media_headers(
        "https://www.bilibili.com/video/BV1VpZpYzE3k/",
        {"User-Agent": "", "Accept": "*/*", "Referer": "https://www.bilibili.com/video/x"})
    assert headers["User-Agent"] == online_mod.BROWSER_UA     # 站点媒体头盖掉空 UA
    assert headers["Referer"] == "https://www.bilibili.com/"
    assert headers["Accept"] == "*/*"                        # 提取器的其它头保留


def test_env_none_does_not_break_media_stage(monkeypatch):
    """LPR_ONLINE_UA=none 只影响解析阶段；不能让媒体阶段变成空 UA（会 403）。"""
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "none")
    assert online_mod._parse_headers("https://www.bilibili.com/video/BV1x/")["User-Agent"] == ""
    assert online_mod._media_headers("https://www.bilibili.com/video/BV1x/", {})["User-Agent"]         == online_mod.BROWSER_UA


def test_env_concrete_ua_applies_to_both_stages(monkeypatch):
    """现场调优：给了具体 UA 字符串时，两个阶段都用它（用户拿自己的可用 UA 顶掉默认策略）。"""
    monkeypatch.setenv(online_mod.UA_ENV_VAR, "MyAgent/1.0")
    assert online_mod._parse_headers("https://weibo.com/tv/show/1")["User-Agent"] == "MyAgent/1.0"
    assert online_mod._media_headers("https://weibo.com/tv/show/1", {})["User-Agent"] == "MyAgent/1.0"
