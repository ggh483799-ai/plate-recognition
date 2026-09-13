"""视频识别接口测试：上传 → 异步任务 → 轮询 → 产物可访问。计划书 §9.3 / §10.2。

用假流水线 + 真实编码的小视频（8 帧 96×64），不加载真实模型，秒级完成。
覆盖的是**契约**：任务生命周期、事件 JSON、产物 URL 可访问、错误码与清理。
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import service.app as app_mod
from service.jobs import MAX_FRAMES_CAP, VideoJobManager
from src.common.interfaces import PlateResult

client = TestClient(app_mod.app)  # 不触发 lifespan

PLATE = PlateResult(
    plate_no="粤A3333G", plate_color="blue", vehicle_type="car",
    det_score=0.88, rec_score=0.97, bbox=[8.0, 12.0, 60.0, 30.0], cost_ms=9.0,
)


class FakePipeline:
    """任何帧都返回同一个车牌：便于断言去重后恰好 1 个事件、hits == 帧数。"""

    engine = object()
    engine_mode = "engine"

    def run(self, frame):
        return [PLATE]


def _make_video(path, frames: int = 8, size: tuple[int, int] = (96, 64), fps: float = 8.0) -> bytes:
    """用真实编码器造一小段视频，返回字节（顺带验证 VideoReader 能解）。"""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "本机 OpenCV 无 mp4v 编码器"
    for i in range(frames):
        writer.write(np.full((h, w, 3), 40 + i * 20, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def manager(monkeypatch):
    """假流水线 + 真实任务目录（这样 /media 才能真的被访问到），用例结束即丢弃产物。"""
    fake = FakePipeline()
    monkeypatch.setattr(app_mod, "_pipeline", fake)        # 通过就绪检查
    monkeypatch.setattr(app_mod, "_build_video_pipeline", lambda: fake)
    mgr = VideoJobManager(app_mod.JOBS_DIR,
                          pipeline_factory=lambda: app_mod._build_video_pipeline())
    monkeypatch.setattr(app_mod, "_video_manager", mgr)
    yield mgr
    mgr.shutdown(wait=False)


def _post_video(data: bytes, name: str = "clip.mp4", **form):
    payload = {"window_s": "3.0", "max_frames": "0"}
    payload.update(form)
    return client.post("/predict_video",
                       files={"file": (name, data, "video/mp4")},
                       data=payload)


def _wait_done(job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/predict_video/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()["data"]
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"任务 {job_id} 超时未完成")


# ============================================================
# 端到端
# ============================================================

def test_video_job_end_to_end(manager, tmp_path):
    data = _make_video(tmp_path / "clip.mp4", frames=8)
    r = _post_video(data)
    assert r.status_code == 200, r.text
    start = r.json()["data"]
    job_id = start["job_id"]
    assert start["status"] in ("queued", "running")
    assert start["poll_url"] == f"/predict_video/{job_id}"

    body = _wait_done(job_id)
    assert body["status"] == "done", body.get("error")
    assert body["error"] == ""

    # 事件：8 帧同一车牌 → 去重成 1 个事件，hits 反映真实命中次数
    assert len(body["events"]) == 1
    ev = body["events"][0]
    assert ev["plate_no"] == "粤A3333G"
    assert ev["plate_color"] == "blue"
    assert ev["hits"] == 8
    assert ev["shot"] == "shot_001.jpg"

    stats = body["stats"]
    assert stats["frames"] == 8
    assert stats["events"] == 1
    assert stats["shots_saved"] == 1
    assert stats["truncated"] is False
    assert body["progress"]["percent"] == pytest.approx(100.0, abs=0.01)

    # 产物 URL 能真的取到
    urls = body["urls"]
    assert urls["annotated"].endswith("/annotated.mp4")
    csv_r = client.get(urls["csv"])
    assert csv_r.status_code == 200
    assert csv_r.content.startswith(b"\xef\xbb\xbf")            # Excel 中文不乱码
    assert "粤A3333G" in csv_r.content.decode("utf-8-sig")

    shot_r = client.get(f"{urls['shots']}/shot_001.jpg")
    assert shot_r.status_code == 200
    assert shot_r.content[:2] == b"\xff\xd8"                    # JPEG magic

    mp4_r = client.get(urls["annotated"])
    assert mp4_r.status_code == 200
    assert len(mp4_r.content) > 0

    # 清理：从任务列表移除（受限环境下磁盘文件可能删不掉，不强制断言 purged）
    info = manager.discard(job_id)
    assert info is not None and info["job_id"] == job_id
    assert manager.get(job_id) is None
    assert manager.discard("nonexistent") is None


def test_video_annotated_output_is_decodable(manager, tmp_path):
    """标注视频必须能被解码出**同样多的帧数**（防 VideoWriter 静默丢帧）。"""
    data = _make_video(tmp_path / "clip.mp4", frames=6)
    job_id = _post_video(data).json()["data"]["job_id"]
    body = _wait_done(job_id)
    assert body["status"] == "done", body.get("error")

    path = app_mod.JOBS_DIR / job_id / "annotated.mp4"
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()

    assert count == body["stats"]["frames"]                     # 计数 == 可解码帧数
    assert body["stats"]["video_frames_written"] == 6
    manager.discard(job_id)


def test_video_respects_max_frames_and_marks_truncated(manager, tmp_path):
    data = _make_video(tmp_path / "clip.mp4", frames=8)
    job_id = _post_video(data, max_frames="3").json()["data"]["job_id"]
    body = _wait_done(job_id)

    assert body["status"] == "done", body.get("error")
    assert body["stats"]["frames"] == 3
    assert body["stats"]["truncated"] is True                   # 如实上报"没看全"
    manager.discard(job_id)


# ============================================================
# 错误路径
# ============================================================

def test_video_rejects_unknown_suffix(manager):
    r = _post_video(b"whatever", name="notes.txt")
    assert r.status_code == 400
    assert "不支持的视频格式" in r.json()["detail"]


def test_video_rejects_bad_window(manager, tmp_path):
    r = _post_video(_make_video(tmp_path / "c.mp4", frames=2), window_s="99")
    assert r.status_code == 400
    assert "window_s" in r.json()["detail"]


def test_video_requires_ready_model(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "_pipeline", None)
    r = _post_video(_make_video(tmp_path / "c.mp4", frames=2))
    assert r.status_code == 503


def test_video_oversize_cleans_up(manager, monkeypatch, tmp_path):
    """超限上传必须返回 413 且**不留垃圾任务目录**。"""
    monkeypatch.setattr(app_mod, "MAX_VIDEO_BYTES", 1024)
    before = {p.name for p in app_mod.JOBS_DIR.iterdir()}
    r = _post_video(_make_video(tmp_path / "c.mp4", frames=8))
    assert r.status_code == 413
    after = {p.name for p in app_mod.JOBS_DIR.iterdir()}
    assert after == before


def test_video_status_unknown_job_404(manager):
    assert client.get("/predict_video/deadbeefdeadbeef").status_code == 404


def test_video_cancel_unknown_job_404(manager):
    assert client.delete("/predict_video/deadbeefdeadbeef").status_code == 404


def test_clamp_max_frames():
    mgr = VideoJobManager(app_mod.JOBS_DIR, pipeline_factory=lambda: None)
    assert mgr.clamp_max_frames(0) == mgr.default_max_frames        # 0 → 服务端默认
    assert mgr.clamp_max_frames(-5) == mgr.default_max_frames
    assert mgr.clamp_max_frames(300) == 300
    assert mgr.clamp_max_frames(10 ** 9) == MAX_FRAMES_CAP          # 硬上限
    mgr.shutdown(wait=False)


# ============================================================
# 多图批量
# ============================================================

def _jpeg_bytes() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_batch_all_ok(monkeypatch):
    monkeypatch.setattr(app_mod, "_pipeline", FakePipeline())
    r = client.post("/predict_batch", files=[
        ("files", ("a.jpg", _jpeg_bytes(), "image/jpeg")),
        ("files", ("b.jpg", _jpeg_bytes(), "image/jpeg")),
    ])
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["count"] == 2
    assert data["ok_count"] == 2
    assert data["total_plates"] == 2
    assert [it["name"] for it in data["items"]] == ["a.jpg", "b.jpg"]
    assert data["items"][0]["plates"][0]["plate_no"] == "粤A3333G"
    assert data["items"][0]["cost_ms"] >= 0


def test_batch_partial_failure_keeps_going(monkeypatch):
    """一张坏图不影响其余：整批仍 200，坏图在 error 里说明原因。"""
    monkeypatch.setattr(app_mod, "_pipeline", FakePipeline())
    r = client.post("/predict_batch", files=[
        ("files", ("bad.jpg", b"not-an-image", "image/jpeg")),
        ("files", ("good.jpg", _jpeg_bytes(), "image/jpeg")),
    ])
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["count"] == 2
    assert data["ok_count"] == 1
    assert data["total_plates"] == 1
    assert "无法解码" in data["items"][0]["error"]
    assert data["items"][1]["error"] == ""


def test_batch_too_many_files(monkeypatch):
    monkeypatch.setattr(app_mod, "_pipeline", FakePipeline())
    files = [("files", (f"{i}.jpg", _jpeg_bytes(), "image/jpeg"))
             for i in range(app_mod.MAX_BATCH_FILES + 1)]
    r = client.post("/predict_batch", files=files)
    assert r.status_code == 400


def test_batch_requires_ready_model(monkeypatch):
    monkeypatch.setattr(app_mod, "_pipeline", None)
    r = client.post("/predict_batch", files=[("files", ("a.jpg", _jpeg_bytes(), "image/jpeg"))])
    assert r.status_code == 503
