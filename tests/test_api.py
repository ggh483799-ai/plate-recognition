"""API 冒烟测试。计划书 §9.3 / §10.2。

不触发 lifespan（真实模型加载），验证端点契约与降级路径。
"""

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from service.app import app

client = TestClient(app)  # 不触发 lifespan，_pipeline 为 None


def _valid_jpeg_bytes() -> bytes:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_contract():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ver"] == "v1.0"
    assert "uptime" in body
    assert body["gpu"] is False


def test_predict_not_ready_returns_503():
    # 有效图片，但模型未加载（lifespan 未触发），应返回 503 而非崩溃
    r = client.post(
        "/predict", files={"file": ("a.jpg", _valid_jpeg_bytes(), "image/jpeg")}
    )
    assert r.status_code == 503


def test_predict_bad_image_returns_400():
    # 非法图片字节应被格式校验拦截
    r = client.post("/predict", files={"file": ("a.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 400


def test_predict_base64_bad_encoding_returns_400():
    r = client.post("/predict_base64", json={"image_b64": "!!not-base64!!"})
    assert r.status_code == 400


def test_predict_base64_valid_returns_503():
    payload = base64.b64encode(_valid_jpeg_bytes()).decode()
    r = client.post("/predict_base64", json={"image_b64": payload})
    assert r.status_code == 503


def test_predict_serializes_plates_when_ready(monkeypatch):
    """模型就绪时应正确序列化非空结果。

    此前用例只覆盖了空结果与降级路径，非空 plates 的字段校验是盲区
    （PlateItem 要求 cost_ms，缺失会直接 500）。
    """
    import service.app as app_mod
    from src.common.interfaces import PlateResult

    class _FakePipeline:
        def run(self, _img):
            return [
                PlateResult(
                    plate_no="京A12345",
                    plate_color="blue",
                    vehicle_type="car",
                    det_score=0.93,
                    rec_score=0.97,
                    bbox=[10.0, 20.0, 130.0, 60.0],
                    cost_ms=12.5,
                )
            ]

    monkeypatch.setattr(app_mod, "_pipeline", _FakePipeline())
    r = client.post(
        "/predict", files={"file": ("a.jpg", _valid_jpeg_bytes(), "image/jpeg")}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 0
    item = body["data"]["plates"][0]
    assert item["plate_no"] == "京A12345"
    assert item["plate_color"] == "blue"
    assert item["rec_score"] == 0.97
    assert item["cost_ms"] == 12.5


def test_index_page_served():
    """根路径应返回上传识别页。"""
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "识别结果" in r.text


def test_live_page_served():
    """实时流页应可访问（摄像头 / RTSP / 边播边识别）。"""
    r = client.get("/live")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "实时" in r.text
    assert "/stream" in r.text        # 页面确实会去调实时接口


def test_static_demo_image_served():
    """示例图应可通过 /static/ 访问。"""
    r = client.get("/static/demo.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
