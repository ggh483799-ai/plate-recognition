"""兜底引擎（hyperlpr3）与 engine_mode 行为的测试。

分两层：
1. 纯逻辑层：类型→颜色映射、模型目录解析、engine_mode 分支 —— 用 fake 引擎，不需要模型。
2. 真实模型层：用 hyperlpr3 真跑一张蓝牌图 —— 模型未就绪时自动 skip。
"""

import numpy as np
import pytest

from src.common.interfaces import BBox, PlateResult
from src.models.hyperlpr_engine import _COLOR_BY_TYPE, _models_ready, plate_type_to_color
from src.pipeline.lpr_pipeline import LprPipeline

# ============================================================
# 测试替身
# ============================================================


class CountingVehicleDetector:
    """可指定类别表的车辆检测器替身。"""

    def __init__(self, cls=0, score=0.9, names=None):
        self.cls = cls
        self.score = score
        if names is not None:
            self.names = names

    def detect(self, frame):
        return [BBox(100, 100, 500, 500, score=self.score, cls=self.cls)]


class CountingPlateDetector:
    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [BBox(200, 300, 400, 340, score=0.95, cls=0)]
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self.boxes)


class FakeRecognizer:
    def __init__(self, text="京A12345", conf=0.9):
        self.text = text
        self.conf = conf

    def recognize(self, plate_img):
        return self.text

    def recognize_with_confidence(self, plate_img):
        return self.text, self.conf


class FakeEngine:
    """兜底引擎替身：记录被调用次数。"""

    def __init__(self, results=None):
        self.results = results if results is not None else [_plate_result()]
        self.calls = 0

    def recognize_frame(self, frame):
        self.calls += 1
        return [PlateResult(**dict(r.__dict__)) for r in self.results]


def _plate_result(**kw):
    base = dict(
        plate_no="粤A3333G",
        plate_color="blue",
        vehicle_type="",
        det_score=0.86,
        rec_score=0.99,
        bbox=[10.0, 10.0, 100.0, 40.0],
        cost_ms=1.0,
    )
    base.update(kw)
    return PlateResult(**base)


def _frame():
    return np.zeros((600, 800, 3), dtype=np.uint8)


# ============================================================
# 纯逻辑层
# ============================================================


def test_plate_type_to_color_mapping():
    # 键取自 hyperlpr3.common.typedef：BLUE=0 / YELLOW_SINGLE=1 / WHILE_SINGLE=2 /
    # GREEN=3 / BLACK_HK_MACAO=4 / YELLOW_DOUBLE=9；港澳(5~8)与未知(-1) 无对应底色
    assert plate_type_to_color(0) == "blue"
    assert plate_type_to_color(1) == "yellow"
    assert plate_type_to_color(2) == "white"
    assert plate_type_to_color(3) == "green"
    assert plate_type_to_color(4) == "black"
    assert plate_type_to_color(9) == "yellow"
    for unknown in (5, 6, 7, 8, -1, 99):
        assert plate_type_to_color(unknown) == "unknown"


def test_color_map_matches_upstream_constants():
    """漂移守卫：上游若改了车牌类型常量编号，这里会失败。"""
    typedef = pytest.importorskip("hyperlpr3.common.typedef")
    assert _COLOR_BY_TYPE[0] == "blue" and typedef.BLUE == 0
    assert typedef.YELLOW_SINGLE == 1 and typedef.WHILE_SINGLE == 2
    assert typedef.GREEN == 3 and typedef.BLACK_HK_MACAO == 4
    assert typedef.YELLOW_DOUBLE == 9


def test_models_ready_requires_all_four_onnx(tmp_path):
    root = tmp_path / ".hyperlpr3"
    onnx_dir = root / "20230229" / "onnx"
    onnx_dir.mkdir(parents=True)
    names = [
        "y5fu_320x_sim.onnx",
        "y5fu_640x_sim.onnx",
        "rpv3_mdict_160_r3.onnx",
        "litemodel_cls_96x_r1.onnx",
    ]
    for n in names[:3]:
        (onnx_dir / n).write_bytes(b"x")
    assert not _models_ready(root, "20230229"), "缺一个模型就应视为未就绪"
    (onnx_dir / names[3]).write_bytes(b"x")
    assert _models_ready(root, "20230229")


# ============================================================
# engine_mode 分支（fake 引擎）
# ============================================================


def test_invalid_engine_mode_raises():
    with pytest.raises(ValueError):
        LprPipeline(
            CountingVehicleDetector(), CountingPlateDetector(), FakeRecognizer(),
            engine=FakeEngine(), engine_mode="bogus",
        )


def test_no_engine_forces_cascade_mode():
    # 未提供引擎时，即使传了 auto 也退化为 cascade，且不会报错
    pipe = LprPipeline(
        CountingVehicleDetector(), CountingPlateDetector(), FakeRecognizer(),
        engine_mode="auto",
    )
    assert pipe.engine_mode == "cascade"
    assert len(pipe.run(_frame())) == 1


def test_auto_uses_engine_when_cascade_empty():
    plate_det = CountingPlateDetector(boxes=[])  # 级联检不到车牌（模拟未训练权重）
    engine = FakeEngine()
    pipe = LprPipeline(
        CountingVehicleDetector(), plate_det, FakeRecognizer(),
        engine=engine, engine_mode="auto",
    )
    results = pipe.run(_frame())
    assert engine.calls == 1
    assert [r.plate_no for r in results] == ["粤A3333G"]


def test_auto_keeps_cascade_when_it_has_result():
    engine = FakeEngine()
    pipe = LprPipeline(
        CountingVehicleDetector(), CountingPlateDetector(), FakeRecognizer(),
        engine=engine, engine_mode="auto",
    )
    results = pipe.run(_frame())
    assert engine.calls == 0, "级联有结果时不应调用兜底引擎"
    assert results[0].plate_no == "京A12345"


def test_cascade_mode_never_calls_engine():
    engine = FakeEngine()
    plate_det = CountingPlateDetector(boxes=[])
    pipe = LprPipeline(
        CountingVehicleDetector(), plate_det, FakeRecognizer(),
        engine=engine, engine_mode="cascade",
    )
    assert pipe.run(_frame()) == []
    assert engine.calls == 0


def test_engine_mode_skips_cascade_entirely():
    engine = FakeEngine()
    plate_det = CountingPlateDetector()
    pipe = LprPipeline(
        CountingVehicleDetector(), plate_det, FakeRecognizer(),
        engine=engine, engine_mode="engine",
    )
    results = pipe.run(_frame())
    assert plate_det.calls == 0, "engine 模式不应跑级联的车牌检测"
    assert engine.calls == 1
    assert results[0].plate_no == "粤A3333G"


def test_engine_result_filtered_by_thresholds():
    low_det = FakeEngine([_plate_result(det_score=0.30)])       # 低于 min_det_score
    pipe = LprPipeline(CountingVehicleDetector(), CountingPlateDetector(boxes=[]),
                       FakeRecognizer(), engine=low_det, engine_mode="engine")
    assert pipe.run(_frame()) == []

    low_rec = FakeEngine([_plate_result(rec_score=0.20)])       # 低于 min_rec_score
    pipe = LprPipeline(CountingVehicleDetector(), CountingPlateDetector(boxes=[]),
                       FakeRecognizer(), engine=low_rec, engine_mode="engine")
    assert pipe.run(_frame()) == []


# ============================================================
# 车辆类别归属（修复「COCO 预训练把 car 说成 truck」）
# ============================================================


def test_vehicle_type_uses_detector_class_names():
    # COCO 预训练：cls=2 的类别名是 car；若硬编码自训练映射会得到 truck
    coco_like = CountingVehicleDetector(cls=2, names={0: "person", 2: "car", 7: "truck"})
    pipe = LprPipeline(coco_like, CountingPlateDetector(boxes=[]),
                       FakeRecognizer(), engine=FakeEngine(), engine_mode="engine")
    assert pipe.run(_frame())[0].vehicle_type == "car"


def test_non_vehicle_class_ignored():
    # 检出的是行人，不应被当作车辆归属；无车辆命中时默认 car
    person_only = CountingVehicleDetector(cls=0, names={0: "person", 2: "car"})
    pipe = LprPipeline(person_only, CountingPlateDetector(boxes=[]),
                       FakeRecognizer(), engine=FakeEngine(), engine_mode="engine")
    assert pipe.run(_frame())[0].vehicle_type == "car"


def test_fallback_mapping_without_names():
    # 检测器无类别表（如测试替身）→ 退回自训练类别序
    pipe = LprPipeline(CountingVehicleDetector(cls=2), CountingPlateDetector(boxes=[]),
                       FakeRecognizer(), engine=FakeEngine(), engine_mode="engine")
    assert pipe.run(_frame())[0].vehicle_type == "truck"


# ============================================================
# 真实模型端到端（模型/图片缺失时 skip）
# ============================================================


def _real_engine_ready() -> bool:
    try:
        from src.models.hyperlpr_engine import ensure_models

        ensure_models(download=False)
        return True
    except Exception:
        return False


_REAL_IMG = __import__("pathlib").Path(__file__).resolve().parents[1] / "data/field/test_car.png"

requires_real = pytest.mark.skipif(
    not (_real_engine_ready() and _REAL_IMG.is_file()),
    reason="hyperlpr3 模型或测试图未就绪",
)


@requires_real
def test_real_engine_reads_blue_plate():
    """真实 hyperlpr3 模型读蓝牌图：车牌号、底色、两个置信度都必须来自模型。"""
    from src.io.reader import imread_bgr
    from src.models.hyperlpr_engine import HyperLprEngine

    img = imread_bgr(_REAL_IMG)
    assert img is not None
    engine = HyperLprEngine(detect_level="high")
    results = engine.recognize_frame(img)

    assert results, "真实模型应至少读出一个车牌"
    top = results[0]
    assert top.plate_no == "粤A3333G", f"识别错误: {top.plate_no}"
    assert top.plate_color == "blue"
    assert 0 < top.det_score <= 1
    assert 0 < top.rec_score <= 1
    assert len(top.bbox) == 4
    assert top.cost_ms > 0
