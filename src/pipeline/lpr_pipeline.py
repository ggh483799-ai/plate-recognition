"""全链路车牌识别流水线（编排层，不写算法）。

计划书 §3.1 数据流：车辆检测 → 逐车裁剪 → 车牌检测 → 透视矫正 → 字符识别 → 后处理。
依赖 Detector / Recognizer 协议注入，便于替换与单测。

两条推理路径
------------
1. **自研级联（cascade）**：车辆检测 → 车牌检测 → 透视矫正 → LPRNet → 规则校验。
   依赖 M2 训练产出的 `plate_best.pt` / `lprnet_best.pt`；权重就绪前拿不到真实车牌号。
2. **兜底引擎（engine）**：现成的 hyperlpr3 ONNX 模型（检测+识别+分类三合一），
   见 `src/models/hyperlpr_engine.py`。权重就绪前保证端到端能出真实车牌号。

`engine_mode` 控制两者关系：
   - `cascade`：只用自研级联（引擎为 None 时的行为，与历史版本一致）
   - `auto`   ：先跑级联，级联无结果时用引擎兜底（默认）
   - `engine` ：只用兜底引擎
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np

from src.common.interfaces import BBox, Detector, PlateResult, Recognizer
from src.postprocess import rules
from src.vision import color as color_mod
from src.vision.warp import warp_plate

log = logging.getLogger(__name__)

# 自训练模型的车辆类别序（0=car, 1=bus, 2=truck, 3=motorcycle），与训练配置一致。
# 仅在检测器拿不到自带类别表（如测试用 fake 检测器）时兜底。
VEHICLE_NAMES = {0: "car", 1: "bus", 2: "truck", 3: "motorcycle"}
# 判定「这个类别属于机动车」的类别名集合（用 COCO 预训练权重时同样适用）
VEHICLE_CLASS_NAMES = {"car", "bus", "truck", "motorcycle"}

# engine_mode 合法取值
ENGINE_MODES = ("auto", "cascade", "engine")


class LprPipeline:
    """三级级联流水线：车辆检测器 + 车牌检测器 + 字符识别器（可选兜底引擎）。"""

    def __init__(
        self,
        vehicle_detector: Detector,
        plate_detector: Detector,
        recognizer: Recognizer,
        min_det_score: float = 0.5,
        min_rec_score: float = 0.6,
        engine=None,
        engine_mode: str = "auto",
        cascade_skip_after: int = 20,
    ):
        if engine_mode not in ENGINE_MODES:
            raise ValueError(f"engine_mode 只能是 {ENGINE_MODES}，收到 {engine_mode!r}")

        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.recognizer = recognizer
        self.min_det_score = min_det_score
        self.min_rec_score = min_rec_score
        # 兜底引擎：需提供 recognize_frame(frame) -> list[PlateResult]
        self.engine = engine
        self.engine_mode = engine_mode if engine is not None else "cascade"

        # 诊断侧信道（见 run() 的 docstring）：每帧被覆盖，供实时页可视化读取
        self.last_vehicles: list[tuple[BBox, str]] = []
        self.last_persons: list[BBox] = []
        self.last_rejected: list[tuple[PlateResult, str]] = []

        # 级联「连续空手」自动跳过：auto 模式下每帧先跑级联再兜底。当级联权重尚未训练
        # （plate 检出恒 0）时这是每帧 ~50ms 的纯浪费。连续 `cascade_skip_after` 帧
        # 级联无任何结果就跳过级联直接走引擎（记一条 WARNING，训练权重换上后自然恢复——
        # 真模型不会连续几十帧一个车牌都检不出）。设 0 关闭该行为。
        self.cascade_skip_after = max(0, int(cascade_skip_after))
        self._cascade_empty_run = 0
        self._cascade_skipped = False

    # --------------------------------------------------------
    def run(self, frame: np.ndarray) -> list[PlateResult]:
        """对单帧执行全链路，返回车牌结果列表。

        诊断侧信道（**不属于返回值**，供实时页可视化；每帧覆盖）：
            `last_vehicles`  [(BBox, 名称)]           检到的机动车
            `last_persons`   [BBox]                   检到的行人
            `last_rejected`  [(PlateResult, 原因)]    被阈值过滤掉的车牌候选
        页面把这几组也画出来，"为什么没识别出这块牌"就从黑盒变成可见的了。
        """
        t0 = time.perf_counter()
        # 车辆/行人检测只做一次：两条路径共用（引擎模式需要它归属车辆类型，级联模式需要它当输入），
        # 可视化层也用同一份结果，**不做第二次推理**（CPU 上多一次 YOLO 就是几百毫秒）。
        vehicles, persons = self._detect_objects(frame)
        self.last_vehicles = [(v, self._vehicle_name(v.cls)) for v in vehicles]
        self.last_persons = list(persons)
        self.last_rejected = []

        if self.engine_mode == "engine":
            results = self._run_engine(frame, vehicles)
            source = "engine"
        else:
            if self._cascade_skipped:
                results = []
                source = "skipped"
            else:
                results = self._run_cascade(frame, vehicles)
                source = "cascade"
            if not results and self.engine is not None and self.engine_mode == "auto":
                if source == "cascade":
                    self._cascade_empty_run += 1
                    if (self.cascade_skip_after
                            and self._cascade_empty_run >= self.cascade_skip_after
                            and not self._cascade_skipped):
                        self._cascade_skipped = True
                        log.warning(
                            "[Pipeline] 级联连续 %d 帧无结果，后续跳过级联直接走兜底引擎"
                            "（换上训练权重后会自动恢复）", self._cascade_empty_run)
                results = self._run_engine(frame, vehicles)
                source = "engine" if source != "skipped" else source
            elif results and source == "cascade":
                self._cascade_empty_run = 0

        results.sort(key=lambda r: r.det_score, reverse=True)
        log.info("[Pipeline] 帧处理完成: 来源=%s, 车牌 %d 个, 总耗时 %.1fms",
                 source, len(results), (time.perf_counter() - t0) * 1000)
        return results

    # --------------------------------------------------------
    def _run_cascade(self, frame: np.ndarray, vehicles: list[BBox]) -> list[PlateResult]:
        """自研三级级联路径。"""
        results: list[PlateResult] = []

        plates = [p for p in self.plate_detector.detect(frame) if p.score >= self.min_det_score]
        log.info("[Pipeline] 级联: 车辆 %d 个, 车牌 %d 个", len(vehicles), len(plates))

        for plate in plates:
            veh = self._nearest_vehicle(plate, vehicles)
            res = self._recognize_one(frame, plate, veh)
            if res is not None:
                results.append(res)
        return results

    def _run_engine(self, frame: np.ndarray, vehicles: list[BBox]) -> list[PlateResult]:
        """兜底引擎路径：模型自带检测+识别，车辆类型用车辆检测器归属。

        被阈值过滤掉的候选会记进 `last_rejected`（可视化层画成虚线框）——
        "检测到了但置信度不够"必须可见，否则用户只会看到黑盒。
        """
        raw = self.engine.recognize_frame(frame)

        kept: list[PlateResult] = []
        rejected: list[tuple[PlateResult, str]] = []
        for r in raw:
            if r.det_score < self.min_det_score:
                reason = f"检测置信度低 {float(r.det_score):.2f} < {self.min_det_score:.2f}"
                log.debug("[Pipeline] 引擎结果丢弃（%s）", reason)
                rejected.append((r, reason))
                continue
            if r.rec_score < self.min_rec_score:
                reason = f"识别置信度低 {float(r.rec_score):.2f} < {self.min_rec_score:.2f}"
                log.debug("[Pipeline] 引擎结果丢弃（%s）", reason)
                rejected.append((r, reason))
                continue
            veh = self._nearest_by_center(
                (r.bbox[0] + r.bbox[2]) / 2, (r.bbox[1] + r.bbox[3]) / 2, vehicles
            )
            r.vehicle_type = self._vehicle_name(veh.cls) if veh else "car"
            kept.append(r)
        self.last_rejected = rejected
        log.info("[Pipeline] 引擎: 车辆 %d 个, 原始车牌 %d 个, 过滤后 %d 个（弃 %d）",
                 len(vehicles), len(raw), len(kept), len(rejected))
        return kept

    # --------------------------------------------------------
    @staticmethod
    def _nearest_by_center(cx: float, cy: float, vehicles: list[BBox]) -> BBox | None:
        """找中心点最近的车辆框。"""
        if not vehicles:
            return None

        def dist(v: BBox) -> float:
            return (cx - (v.x1 + v.x2) / 2) ** 2 + (cy - (v.y1 + v.y2) / 2) ** 2

        return min(vehicles, key=dist)

    @staticmethod
    def _nearest_vehicle(plate: BBox, vehicles: list[BBox]) -> BBox | None:
        """找车牌中心最近的车辆框（用于归属车辆类型）。"""
        cx = (plate.x1 + plate.x2) / 2
        cy = (plate.y1 + plate.y2) / 2
        return LprPipeline._nearest_by_center(cx, cy, vehicles)

    def _detect_objects(self, frame: np.ndarray) -> tuple[list[BBox], list[BBox]]:
        """一次推理，把检测结果拆成「机动车」与「行人」两组。

        行人不属于机动车过滤范围，是可视化层单独要的——但绝不能为此再跑一次检测
        （CPU 上多一次 YOLO 就是几百毫秒）。
        """
        vehicles: list[BBox] = []
        persons: list[BBox] = []
        for det in self.vehicle_detector.detect(frame):
            if det.score < self.min_det_score:
                continue
            name = self._vehicle_name(det.cls)
            if name in VEHICLE_CLASS_NAMES:
                vehicles.append(det)
            elif name == "person":
                persons.append(det)
        return vehicles, persons

    def _vehicle_boxes(self, frame: np.ndarray) -> list[BBox]:
        """车辆检测：按置信度 + 「属于机动车类别」双重过滤。

        不过滤类别的话，COCO 预训练权重检出的行人 / 交通标志也可能被当成「最近车辆」。
        """
        return self._detect_objects(frame)[0]

    def _vehicle_name(self, cls: int) -> str:
        """按检测器自带类别表解析车辆类型名。

        预训练 COCO 与自训练模型的类别序不同（COCO 里 cls=2 是 car，自训练 4 类里 cls=2
        是 truck），硬编码映射会把轿车说成卡车，所以优先用模型自带类别表。
        """
        names = getattr(self.vehicle_detector, "names", None)
        if isinstance(names, dict) and cls in names:
            return str(names[cls])
        return VEHICLE_NAMES.get(cls, "car")

    def _is_vehicle(self, cls: int) -> bool:
        """该类别是否属于机动车。检测器无类别表时退回自训练类别序判断。"""
        names = getattr(self.vehicle_detector, "names", None)
        if isinstance(names, dict) and cls in names:
            return str(names[cls]) in VEHICLE_CLASS_NAMES
        return cls in VEHICLE_NAMES

    def _recognize_one(self, frame: np.ndarray, plate: BBox, veh: BBox | None) -> PlateResult | None:
        """识别单个车牌：透视矫正 → 字符识别 → 后处理校验。"""
        t0 = time.perf_counter()
        # 透视矫正到 94×24
        pts = self._bbox_to_pts(plate)
        plate_img = warp_plate(frame, pts)

        # 字符识别（取识别器给出的真实置信度）
        raw_text, rec_score = self._recognize_text(plate_img)

        # 置信度不足直接丢弃（识别器未提供置信度时该检查不生效）
        if rec_score > 0 and rec_score < self.min_rec_score:
            log.debug("[PostRules] 识别置信度过低: %.3f < %.3f", rec_score, self.min_rec_score)
            return None

        plate_no = rules.correct_confusable(raw_text)
        ok, reason = rules.validate_plate(plate_no)

        if not ok:
            log.debug("[PostRules] 车牌校验失败: %r 原因=%s", plate_no, reason)
            # 仍尝试用正则兜底提取
            plate_no = rules.extract_plate_candidate(plate_no) or plate_no
            ok, _ = rules.validate_plate(plate_no)
            if not ok:
                return None

        return PlateResult(
            plate_no=plate_no,
            plate_color=color_mod.plate_color(plate_img),
            vehicle_type=self._vehicle_name(veh.cls) if veh else "car",
            det_score=plate.score,
            rec_score=rec_score,
            bbox=plate.xyxy,
            cost_ms=(time.perf_counter() - t0) * 1000,
        )

    def _recognize_text(self, plate_img: np.ndarray) -> tuple[str, float]:
        """调用识别器取文本与置信度。

        Recognizer 协议只要求 recognize()，因此对支持
        recognize_with_confidence() 的实现优先取真实置信度，否则退回 0.0。
        """
        rich = getattr(self.recognizer, "recognize_with_confidence", None)
        if callable(rich):
            return rich(plate_img)
        return self.recognizer.recognize(plate_img), 0.0

    @staticmethod
    def _bbox_to_pts(plate: BBox) -> np.ndarray:
        """矩形框转四角点（无角点信息时用外接矩形，计划书 §4.3）。"""
        x1, y1, x2, y2 = plate.xyxy
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


# ============================================================
# 构建与 CLI
# ============================================================

def build_engine(cfg: dict, detect_level: str | None = None):
    """按配置构建兜底引擎；依赖缺失或模型不可用时返回 None（降级不报错）。

    `detect_level` 显式传入时覆盖配置（实时流场景要 low=320px 换速度，见 `src/pipeline/stream.py`）。
    注意**不能直接改 cfg**：`load_config` 带 lru_cache，返回的是同一个 dict 对象，
    改它等于全局改配置且不可回滚。
    """
    try:
        from src.models.hyperlpr_engine import HyperLprEngine

        return HyperLprEngine(
            detect_level=detect_level or cfg.get("engine_detect_level", "high"),
            logger_level=cfg.get("engine_logger_level", 3),
        )
    except Exception as exc:  # hyperlpr3 未装 / 模型下载失败 → 退回纯级联
        log.warning("[Pipeline] 兜底引擎不可用，仅使用自研级联: %s", exc)
        return None


def build_pipeline(engine_level: str | None = None,
                   min_det_score: float | None = None,
                   min_rec_score: float | None = None) -> LprPipeline:
    """按 configs/lprnet.yaml 构建流水线（服务与 CLI 共用）。

    覆盖参数（都可选，其余仍读配置）：
        `engine_level`   兜底引擎检测档位（low/high），实时流用 low 换速度
        `min_det_score`  车牌检测阈值：**玩具车 / 小车牌**经常低于默认 0.5，调低才有机会被识别
        `min_rec_score`  字符识别阈值：玩具牌的字符样式不在训练分布里，识别置信度天然偏低
    """
    from src.common.config import load_config, weight_path
    from src.models.detector import YoloDetector
    from src.models.lpr_recognizer import LprNetRecognizer

    cfg = load_config("lprnet")
    device = cfg.get("device", "cpu")
    vehicle = YoloDetector(str(weight_path(cfg.get("vehicle_weights", "yolov8n.pt"))), device=device)
    plate = YoloDetector(str(weight_path(cfg.get("plate_weights", "yolov8n.pt"))), device=device)
    recognizer = LprNetRecognizer(str(weight_path(cfg.get("lprnet_weights", "lprnet_best.pt"))), device=device)

    engine = None
    if cfg.get("engine_mode", "auto") != "cascade":
        engine = build_engine(cfg, detect_level=engine_level)

    return LprPipeline(
        vehicle,
        plate,
        recognizer,
        min_det_score=cfg.get("min_det_score", 0.5) if min_det_score is None else float(min_det_score),
        min_rec_score=cfg.get("min_rec_score", 0.6) if min_rec_score is None else float(min_rec_score),
        engine=engine,
        engine_mode=cfg.get("engine_mode", "auto"),
    )


def main() -> int:
    """CLI 入口：python -m src.pipeline.lpr_pipeline --img demo.jpg"""
    from src.common.logger import setup_logger
    from src.io.reader import imread_bgr
    setup_logger("lpr_pipeline")

    parser = argparse.ArgumentParser(description="车牌识别全链路（单图）")
    parser.add_argument("--img", required=True, help="输入图片路径")
    parser.add_argument("--device", default="cpu", help="cpu / 0(GPU)，仅影响自研级联")
    parser.add_argument("--engine-mode", choices=ENGINE_MODES, default=None,
                        help="auto(默认) / cascade(仅自研) / engine(仅兜底引擎)")
    args = parser.parse_args()

    img = imread_bgr(args.img)
    if img is None:
        logging.error("[Pipeline] 无法读取图片: %s", args.img)
        return 1

    if args.device != "cpu":
        # 命令行显式指定 device 时，覆盖配置
        from src.common.config import load_config
        load_config("lprnet")["device"] = args.device

    pipeline = build_pipeline()
    if args.engine_mode is not None:
        pipeline.engine_mode = args.engine_mode if pipeline.engine else "cascade"

    results = pipeline.run(img)
    output = {
        "code": 0,
        "msg": "success",
        "data": {"plates": [r.to_dict() for r in results]},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
