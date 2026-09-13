"""Pipeline 集成测试（依赖注入 fake 检测器/识别器）。计划书 §9.3。"""

import numpy as np
import pytest

from src.common.interfaces import BBox, PlateResult
from src.pipeline.lpr_pipeline import LprPipeline


class FakeVehicleDetector:
    def detect(self, frame):
        return [BBox(100, 100, 500, 500, score=0.9, cls=0)]


class FakePlateDetector:
    def detect(self, frame):
        return [BBox(200, 300, 400, 340, score=0.95, cls=0)]


class FakeRecognizer:
    def recognize(self, plate_img):
        return "京A12345"


class EmptyDetector:
    def detect(self, frame):
        return []


def _frame():
    return np.zeros((600, 800, 3), dtype=np.uint8)


def test_pipeline_end_to_end():
    pipeline = LprPipeline(FakeVehicleDetector(), FakePlateDetector(), FakeRecognizer())
    results = pipeline.run(_frame())
    assert len(results) == 1
    r = results[0]
    assert r.plate_no == "京A12345"
    assert r.vehicle_type == "car"
    assert r.det_score == pytest.approx(0.95)


def test_pipeline_plate_only_no_vehicle():
    # 有车牌无车辆：仍识别，车辆类型默认 car（车牌特写/摩托车场景）
    pipeline = LprPipeline(EmptyDetector(), FakePlateDetector(), FakeRecognizer())
    results = pipeline.run(_frame())
    assert len(results) == 1
    assert results[0].vehicle_type == "car"


def test_pipeline_no_detection():
    # 车辆和车牌都为空 → 空结果
    pipeline = LprPipeline(EmptyDetector(), EmptyDetector(), FakeRecognizer())
    results = pipeline.run(_frame())
    assert results == []


class FakeCocoDetector:
    """模拟 COCO 预训练权重的类别表：0=person, 2=car, 5=bus, 7=truck。"""

    names = {0: "person", 2: "car", 5: "bus", 7: "truck"}

    def __init__(self, dets):
        self._dets = list(dets)

    def detect(self, frame):
        return list(self._dets)


def test_pipeline_exposes_vehicles_and_persons():
    """车辆/行人要暴露给可视化层：一次推理拆两组，行人不许混进车辆。"""
    dets = [
        BBox(10, 10, 200, 400, score=0.8, cls=0),     # person
        BBox(300, 100, 700, 400, score=0.9, cls=2),   # car
        BBox(0, 0, 50, 50, score=0.4, cls=2),         # 低于阈值 → 不出现
    ]
    pipeline = LprPipeline(FakeCocoDetector(dets), EmptyDetector(), FakeRecognizer(),
                           min_det_score=0.5)
    pipeline.run(_frame())

    assert [(name, round(b.score, 3)) for b, name in pipeline.last_vehicles] == [("car", 0.9)]
    assert [round(b.score, 3) for b in pipeline.last_persons] == [0.8]


def test_pipeline_collects_rejected_candidates():
    """被阈值过滤的车牌候选必须可见——否则"为什么没识别出这块牌"就是个黑盒。"""

    class FakeEngine:
        def recognize_frame(self, frame):
            return [
                PlateResult(plate_no="粤A3333G", det_score=0.31, rec_score=0.42, bbox=[1, 2, 30, 12]),
                PlateResult(plate_no="京A12345", det_score=0.9, rec_score=0.95, bbox=[40, 2, 70, 12]),
            ]

    pipeline = LprPipeline(EmptyDetector(), EmptyDetector(), FakeRecognizer(),
                           engine=FakeEngine(), engine_mode="engine",
                           min_det_score=0.5, min_rec_score=0.6)
    results = pipeline.run(_frame())

    assert [r.plate_no for r in results] == ["京A12345"]
    assert len(pipeline.last_rejected) == 1
    rejected, reason = pipeline.last_rejected[0]
    assert rejected.plate_no == "粤A3333G"
    assert "检测置信度低" in reason


def test_build_pipeline_threshold_override():
    """阈值要能按实例覆盖（玩具车/小车牌需要放宽），且不动全局配置。"""
    from src.pipeline.lpr_pipeline import build_pipeline

    p = build_pipeline(min_det_score=0.15, min_rec_score=0.2)
    assert p.min_det_score == pytest.approx(0.15)
    assert p.min_rec_score == pytest.approx(0.2)
    # 默认构建仍是配置里的 0.5/0.6
    d = build_pipeline()
    assert d.min_det_score == pytest.approx(0.5)
    assert d.min_rec_score == pytest.approx(0.6)


def test_pipeline_invalid_plate_dropped():
    # 识别器返回非法车牌号，应被后处理规则过滤
    class BadRecognizer:
        def recognize(self, plate_img):
            return "XXXXXXX"

    pipeline = LprPipeline(FakeVehicleDetector(), FakePlateDetector(), BadRecognizer())
    results = pipeline.run(_frame())
    assert results == []
