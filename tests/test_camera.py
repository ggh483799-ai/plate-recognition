"""浏览器摄像头链路测试：会话 ingest 语义、并发/节流、事件与截图、管理器回收、接口契约。

背景（2026-09-14 真实事故）：页面部署到云服务器后，「本机摄像头」调的是 `/stream/devices`
——那是**服务器**上的摄像头，云服务器没接设备，用户看到"未检测到可用摄像头"。
本链路把"本机摄像头"改成**浏览者自己电脑的摄像头**：浏览器采集 → 推帧 → 服务端只回识别结果。

全部用注入的 fake pipeline 与合成 JPEG，不加载真实模型、不依赖摄像头，秒级跑完。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import service.app as app_mod
from service.cameras import (
    MEDIA_BASE_PREFIX,
    CameraSessionManager,
    TooManyCameras,
)
from src.common.interfaces import BBox, PlateResult
from src.pipeline.browser_camera import ClientCameraSession
from src.pipeline.stream import STATUS_RUNNING, STATUS_STOPPED

client = TestClient(app_mod.app)  # 不触发 lifespan

PLATE = PlateResult(
    plate_no="粤A3333G", plate_color="blue", vehicle_type="car",
    det_score=0.88, rec_score=0.97, bbox=[8.0, 12.0, 60.0, 30.0], cost_ms=9.0,
)


class FakePipeline:
    """可控流水线：可注入结果，也能"卡住"以复现并发丢帧。"""

    def __init__(self, results=None, hold: threading.Event | None = None,
                 started: threading.Event | None = None):
        self.results = results if results is not None else [PLATE]
        self.calls = 0
        self.last_shape: tuple[int, int] | None = None
        self._hold = hold
        self._started = started
        self.last_vehicles: list = [(BBox(1, 2, 30, 40, 0.81, 2), "car")]  # (BBox, 类别名)
        self.last_persons: list = []
        self.last_rejected: list = []

    def run(self, frame):
        self.calls += 1
        self.last_shape = (int(frame.shape[1]), int(frame.shape[0]))
        if self._started is not None:
            self._started.set()
        if self._hold is not None:
            self._hold.wait(timeout=5.0)
        return list(self.results)


def _jpeg(width: int = 320, height: int = 180, value: int = 120) -> bytes:
    """造一张真实可解码的 JPEG（走 cv2，确保和线上同一条解码路径）。"""
    img = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    assert ok
    return buf.tobytes()


def _session(tmp_path=None, *, pipeline=None, **kw):
    opts = dict(interval_ms=0, max_side=640, idle_timeout_s=0.0,
                media_base_prefix=MEDIA_BASE_PREFIX)
    opts.update(kw)
    return ClientCameraSession(
        "cam-test",
        pipeline or FakePipeline(),
        out_dir=(tmp_path / "shots_root") if tmp_path is not None else None,
        **opts,
    )


# ============================================================
# 会话：解码与结果
# ============================================================

def test_ingest_returns_plate_boxes_and_stats():
    session = _session()
    payload = session.ingest(_jpeg())

    assert payload["status"] == STATUS_RUNNING
    assert payload["reused"] is False
    assert payload["frame_size"] == [320, 180]          # box 坐标所在空间
    assert [b["plate_no"] for b in payload["plates"]] == ["粤A3333G"]
    assert payload["plates"][0]["bbox"] == [8.0, 12.0, 60.0, 30.0]
    assert payload["stats"]["frames"] == 1
    assert payload["stats"]["recog_runs"] == 1
    # 车辆侧信道也要出来（前端画细线框）
    assert payload["objects"] and payload["objects"][0]["label"] == "car"


def test_ingest_rejects_garbage_and_empty():
    session = _session()
    for bad in (b"", b"not-an-image", b"\x00\x01\x02"):
        with pytest.raises(ValueError):
            session.ingest(bad)
    assert session.recog_runs == 0          # 坏帧不能被算成一次识别


def test_ingest_downscales_to_max_side():
    """上传尺寸由客户端决定，服务端不信任它：长边必须收敛到 max_side 以内。"""
    pipeline = FakePipeline()
    session = _session(pipeline=pipeline, max_side=320)
    payload = session.ingest(_jpeg(width=1600, height=900, value=90))

    assert payload["frame_size"] == [320, 180]
    assert pipeline.last_shape == (320, 180)


def test_ingest_throttles_to_interval():
    """间隔内的帧沿用上次结果（reused=true），不重复推理——客户端推得快也不加重服务端。"""
    pipeline = FakePipeline()
    session = _session(pipeline=pipeline, interval_ms=60000)   # 60s 内不会二次推理
    first = session.ingest(_jpeg(value=100))
    second = session.ingest(_jpeg(value=130))

    assert first["reused"] is False
    assert second["reused"] is True
    assert pipeline.calls == 1
    assert second["stats"]["recog_runs"] == 1
    assert second["stats"]["skipped"] == 1


def test_ingest_drops_frame_while_busy():
    """推理进行中到来的帧直接丢弃并计数（只认最新帧，不排队攒延迟）。"""
    hold, started = threading.Event(), threading.Event()
    session = _session(pipeline=FakePipeline(hold=hold, started=started))

    worker = threading.Thread(target=session.ingest, args=(_jpeg(value=60),))
    worker.start()
    assert started.wait(timeout=3.0), "第一次推理没有开始"

    # 此刻推理被 hold 卡在 pipeline.run 里（run_lock 被占），这一帧必须走"丢弃"分支
    busy = session.ingest(_jpeg(value=200))
    assert busy["reused"] is True

    hold.set()                       # 放行第一次推理，等它正常收尾
    worker.join(timeout=5.0)

    assert session.recog_dropped == 1
    assert session.recog_runs == 1   # 只有第一帧真的跑了推理


def test_ingest_dedups_events_and_writes_shots(tmp_path):
    """同车牌多帧 → 1 个事件、多次命中，小图落盘且地址前缀正确。"""
    session = _session(tmp_path)
    session.ingest(_jpeg(value=80))
    session.ingest(_jpeg(value=120))

    snap = session.snapshot()
    assert snap["stats"]["events"] == 1
    assert snap["stats"]["total_hits"] == 2
    assert snap["events"][0]["plate_no"] == "粤A3333G"
    assert snap["events"][0]["shot"]
    assert snap["media_base"].startswith(MEDIA_BASE_PREFIX)
    shot = tmp_path / "shots_root" / "shots" / snap["events"][0]["shot"]
    assert shot.is_file() and shot.stat().st_size > 0


def test_snapshot_shape_matches_stream_session():
    """结构必须与拉流会话一致：前端同一套 KPI 面板与 StreamStatusData 模型直接复用。"""
    snap = _session().snapshot()
    for key in ("session_id", "status", "source", "error", "stats", "events", "media_base", "source_title"):
        assert key in snap, key
    for key in ("frames", "read_frames", "recog_runs", "recog_dropped", "avg_recog_ms",
                "events", "total_hits", "elapsed_s", "push_fps", "recog_fps", "interval_ms",
                "max_side", "live", "backend", "skipped"):
        assert key in snap["stats"], key
    assert snap["stats"]["live"] is True
    assert snap["stats"]["backend"] == "浏览器推帧"


def test_stop_is_idempotent_and_idle_grows():
    session = _session()
    session.ingest(_jpeg())
    session.stop()
    session.stop()                      # 重复调用不该抛
    assert session.status == STATUS_STOPPED
    assert session.is_finished is True
    assert session.idle_seconds() >= 0


# ============================================================
# 管理器：并发上限、空闲回收、产物
# ============================================================

def _manager(tmp_path, **kw):
    opts = dict(max_sessions=2, idle_timeout_s=3600.0, ttl_s=3600.0)
    opts.update(kw)
    return CameraSessionManager(tmp_path / "cameras", pipeline_factory=lambda lvl, d, r: FakePipeline(), **opts)


def test_manager_caps_concurrent_sessions(tmp_path):
    mgr = _manager(tmp_path, max_sessions=1)
    mgr.create("s1")
    with pytest.raises(TooManyCameras):
        mgr.create("s2")


def test_manager_reaps_idle_session(tmp_path):
    """浏览器推帧没有长连接，只能按"多久没收到帧"回收（用户关页面后把算力位让出来）。"""
    mgr = _manager(tmp_path, max_sessions=1, idle_timeout_s=3600.0)
    session = mgr.create("s1")
    session.idle_timeout_s = 0.0        # 把这一路的空闲阈值调成 0，等价于"用户已经走了"
    assert mgr.expire_idle() == 1
    assert session.status == STATUS_STOPPED

    # 位子被让出来了：同样的会话号能再建一路（并发上限=1）
    mgr.create("s2")
    assert mgr.count()["active"] == 1


def test_manager_ingest_unknown_session_returns_none(tmp_path):
    mgr = _manager(tmp_path)
    assert mgr.ingest("nope", _jpeg()) is None
    assert mgr.snapshot("nope") is None
    assert mgr.stop("nope") is False
    assert mgr.remove("nope") is None


def test_manager_remove_drops_registry_entry(tmp_path):
    mgr = _manager(tmp_path)
    mgr.create("s1")
    info = mgr.remove("s1")
    assert info["session_id"] == "s1"
    assert "purged" in info
    assert mgr.get("s1") is None


# ============================================================
# 接口契约
# ============================================================

@pytest.fixture
def camera_manager(monkeypatch, tmp_path):
    fake = FakePipeline()
    monkeypatch.setattr(app_mod, "_pipeline", fake)     # 通过就绪检查
    mgr = CameraSessionManager(tmp_path / "cameras",
                               pipeline_factory=lambda lvl, d, r: fake,
                               idle_timeout_s=3600.0)
    monkeypatch.setattr(app_mod, "_camera_manager", mgr)
    yield mgr
    mgr.stop_all()


def test_camera_session_requires_ready_model(monkeypatch):
    monkeypatch.setattr(app_mod, "_pipeline", None)
    r = client.post("/camera/session")
    assert r.status_code == 503


def test_camera_session_endpoint_urls(camera_manager):
    r = client.post("/camera/session", data={"interval_ms": "200", "level": "low", "max_side": "640"})
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    sid = d["session_id"]
    assert d["frame_url"] == f"/camera/{sid}/frame"
    assert d["status_url"] == f"/camera/{sid}"
    assert d["media_base"].endswith("/shots")
    assert d["interval_ms"] == 200 and d["max_side"] == 640
    assert "上传" in d["info"]           # 必须向用户说明帧会上传（隐私披露）


def test_camera_frame_round_trip(camera_manager):
    sid = client.post("/camera/session", data={"interval_ms": "0"}).json()["data"]["session_id"]
    r = client.post(f"/camera/{sid}/frame",
                    files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()["data"]

    kinds = {b["kind"] for b in d["boxes"]}
    assert "plate" in kinds and "vehicle" in kinds
    plate_box = [b for b in d["boxes"] if b["kind"] == "plate"][0]
    assert plate_box["label"] == "粤A3333G" and plate_box["plate_color"] == "blue"
    assert d["frame_size"] == [320, 180]
    assert d["stats"]["recog_runs"] == 1
    assert [e["plate_no"] for e in d["events"]] == ["粤A3333G"]


def test_camera_frame_rejects_bad_input(camera_manager):
    sid = client.post("/camera/session").json()["data"]["session_id"]
    # 空帧
    r = client.post(f"/camera/{sid}/frame", files={"file": ("f.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    # 不是图片
    r = client.post(f"/camera/{sid}/frame", files={"file": ("f.jpg", b"junk", "image/jpeg")})
    assert r.status_code == 400
    assert "图片" in r.json()["detail"]
    # 会话不存在
    r = client.post("/camera/deadbeef/frame", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 404


def test_camera_frame_rejects_oversized(camera_manager, monkeypatch):
    monkeypatch.setattr(app_mod, "MAX_CAMERA_FRAME_BYTES", 128)
    sid = client.post("/camera/session").json()["data"]["session_id"]
    r = client.post(f"/camera/{sid}/frame", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 413


def test_camera_status_and_delete(camera_manager):
    sid = client.post("/camera/session").json()["data"]["session_id"]
    client.post(f"/camera/{sid}/frame", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})

    r = client.get(f"/camera/{sid}")
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["stats"]["events"] == 1
    assert d["stats"]["backend"] == "浏览器推帧"

    assert client.delete(f"/camera/{sid}").status_code == 200
    assert client.get(f"/camera/{sid}").json()["data"]["status"] == STATUS_STOPPED
    # 带 purge 的删除会把会话整条移除
    assert client.delete(f"/camera/{sid}", params={"purge": "true"}).status_code == 200
    assert client.get(f"/camera/{sid}").status_code == 404


def test_camera_unknown_session_endpoints_404(camera_manager):
    assert client.get("/camera/deadbeef").status_code == 404
    assert client.delete("/camera/deadbeef").status_code == 404
