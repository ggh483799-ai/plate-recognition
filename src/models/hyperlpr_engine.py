"""现成中文车牌识别引擎（hyperlpr3 / ONNX）。

为什么存在
----------
项目自研的三级级联（车辆检测 → 车牌检测 → 字符识别）依赖 M2 训练产出的
`weights/plate_best.pt` 与 `weights/lprnet_best.pt`。权重产出前，端到端拿不到真实
车牌号：COCO 预训练模型没有 plate 类（车牌检测恒为 0），LPRNet 是随机权重。

本模块把开源 hyperlpr3（Apache-2.0；检测 + 识别 + 分类三合一 ONNX）封装成项目统一的
`PlateResult`，作为 `LprPipeline` 的可选引擎接入，使端到端在训练权重就绪前即可产出
真实车牌号（计划书 §6 R2 降级路径的一个具体实现）。

与自研级联的差异（如实标注，不做粉饰）
------------------------------------
1. **结果不经本项目 rules 硬过滤**：hyperlpr3 是已训练模型，而本地规则表并不完整
   （学 / 警 / 港 / 澳 / 双层牌等未穷举），用规则硬过滤会误杀正确结果。
   本模块只做 `strip()` + 大写归一，并把 `rules.validate_plate` 的结论写进日志供观测。
2. **底色取自模型自带分类器**（type_idx），比 HSV 阈值法更稳。
3. **检测 / 识别两个置信度均取自模型真实输出**（内部 `Plate.dex_bound_confidence` /
   `rec_confidence`），不做任何填充或复用。

依赖说明
--------
`hyperlpr3` 首次使用会从远端下载约 12MB 的 ONNX 模型包。其自带下载函数在解压后调用
`os.remove()` 清理临时 zip，在受限环境（回收站不可用）下会抛 `OSError` 并**中断初始化**。
本模块用 `ensure_models()` 自行完成「下载 → 解压 → 尽力清理」，避开该问题，做到开箱自举。
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np

from src.common.interfaces import BBox, PlateResult
from src.postprocess import rules

log = logging.getLogger(__name__)

# ============================================================
# ONNX Runtime 会话线程数收敛（性能关键，实测数据见下）
# ------------------------------------------------------------
# hyperlpr3 内部用 `ort.InferenceSession(path, None)` 建会话——默认**每个会话开满核
# 线程池**。本进程同时存在 5+ 个会话（hyperlpr3 × 3 + 自研 ONNX 导出），Windows 上
# 这些线程池互相休眠/唤醒，实测：一次 torch YOLO 在 ORT 会话跑过后要多付 **300~450ms
# 的调度税**（d after e: 430ms vs 单独 30ms），流水线单帧从 ~80ms 恶化到 500-600ms。
# 把每个会话收敛到单线程后：交替 5 轮全部稳定在 36-56ms（18 倍改善），且这些小模型
# 单线程推理并不比多线程慢（引擎 9.8ms vs 默认 13ms）。
# 可用环境变量 `LPR_ORT_THREADS` 覆盖（设 0 = 不干预，用 ORT 默认）。
# ============================================================

def _apply_ort_thread_policy() -> None:
    """把 ort.InferenceSession 包一层：默认单线程会话（进程级，一次生效）。

    只在 hyperlpr3 首次真正建会话之前调用一次。用环境变量 LPR_ORT_THREADS 可覆盖。
    """
    try:
        import onnxruntime as ort
        from onnxruntime import SessionOptions
    except ImportError:  # 未装 ORT 时让上游正常报错
        return

    try:
        threads = int(os.environ.get("LPR_ORT_THREADS", "1"))
    except ValueError:
        threads = 1
    if threads <= 0 or getattr(ort.InferenceSession, "_lpr_patched", False):
        return

    _orig = ort.InferenceSession

    def _patched(path, *args, **kwargs):
        opts = SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        kwargs.pop("sess_options", None)          # 统一收敛，忽略调用方自带配置
        return _orig(path, sess_options=opts, **kwargs)

    _patched._lpr_patched = True  # type: ignore[attr-defined]
    ort.InferenceSession = _patched
    log.info("[Engine] ONNX Runtime 会话已收敛为 %d 线程（消除多会话调度税）", threads)


_apply_ort_thread_policy()


# 与 hyperlpr3 内部常量一致，仅在上游文件不可读时兜底（见 _read_upstream_constants）
_FALLBACK_MODEL_VERSION = "20230229"
_FALLBACK_ONLINE_URL = "http://hyperlpr.tunm.top/raw/"

# 期望存在的 4 个 ONNX 模型（检测 320/640、识别、分类）
_REQUIRED_ONNX = (
    "y5fu_320x_sim.onnx",
    "y5fu_640x_sim.onnx",
    "rpv3_mdict_160_r3.onnx",
    "litemodel_cls_96x_r1.onnx",
)

# 车牌类型 idx → 项目底色词（键取自 hyperlpr3.common.typedef：
# BLUE=0 / YELLOW_SINGLE=1 / WHILE_SINGLE=2 / GREEN=3 / BLACK_HK_MACAO=4 / YELLOW_DOUBLE=9）
# 港澳单双层（5~8）与未知（-1）没有对应底色类别 → unknown
_COLOR_BY_TYPE = {0: "blue", 1: "yellow", 2: "white", 3: "green", 4: "black", 9: "yellow"}

_CAPACITY = {  # 检测级别 → (detect_level 常量, 输入尺寸描述)
    "low": 320,
    "high": 640,
}


# ============================================================
# 模型自举（绕开上游 os.remove 在受限环境被拦的问题）
# ============================================================

def _read_upstream_constants() -> tuple[str, str]:
    """读出上游的模型版本号与下载地址。

    用「读文件 + 正则」而不是 `import hyperlpr3`：后者会执行其 `__init__` 里的
    `initialization()`，可能触发下载与 `os.remove`，在受限环境下抛异常。
    """
    try:
        spec = importlib.util.find_spec("hyperlpr3")
        if not spec or not spec.submodule_search_locations:
            raise ImportError("hyperlpr3 未安装")
        settings_py = Path(list(spec.submodule_search_locations)[0]) / "config" / "settings.py"
        text = settings_py.read_text(encoding="utf-8", errors="ignore")
        ver = re.search(r'_MODEL_VERSION_\s*=\s*"([^"]+)"', text)
        url = re.search(r'_ONLINE_URL_\s*=\s*"([^"]+)"', text)
        return (
            ver.group(1) if ver else _FALLBACK_MODEL_VERSION,
            url.group(1) if url else _FALLBACK_ONLINE_URL,
        )
    except Exception as exc:  # 读不到就用兜底常量，不影响主流程
        log.debug("[HyperLpr] 读取上游常量失败，使用兜底值: %s", exc)
        return _FALLBACK_MODEL_VERSION, _FALLBACK_ONLINE_URL


def _model_roots() -> list[Path]:
    """上游模型目录的候选位置（按优先级）。

    上游用 `os.path.join(os.environ['HOMEPATH'], '.hyperlpr3')`，而 Windows 的 HOMEPATH
    不含盘符（形如 `\\Users\\Administrator`），实际落盘位置取决于**当前盘符**。因此这里同时
    枚举「与上游一致的解析结果」「HOMEDRIVE+HOMEPATH」「Path.home()」三种可能。
    """
    cands: list[Path] = []
    homepath = os.environ.get("HOMEPATH")
    if homepath:
        cands.append(Path(os.path.abspath(os.path.join(homepath, ".hyperlpr3"))))
        homedrive = os.environ.get("HOMEDRIVE", "")
        if homedrive:
            cands.append(Path(homedrive + homepath) / ".hyperlpr3")
    cands.append(Path.home() / ".hyperlpr3")

    seen: set[str] = set()
    out: list[Path] = []
    for c in cands:
        key = str(c).lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _models_ready(root: Path, version: str) -> bool:
    """4 个 ONNX 模型是否都已就位。"""
    onnx_dir = root / version / "onnx"
    return all((onnx_dir / name).is_file() for name in _REQUIRED_ONNX)


def _download_and_extract(online_url: str, version: str, target: Path) -> None:
    """下载模型 zip 并解压。临时 zip 的清理是「尽力而为」，失败不影响使用。"""
    import requests

    target.mkdir(parents=True, exist_ok=True)
    url = f"{online_url}{version}.zip"
    zip_path = target / f"{version}.zip"
    log.info("[HyperLpr] 下载模型: %s", url)

    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)

    try:
        zip_path.unlink()
    except OSError as exc:
        # 上游正是在这里直接 os.remove 而中断初始化的；清理失败不影响模型可用
        log.warning("[HyperLpr] 临时 zip 未能删除（不影响使用）: %s", exc)


def ensure_models(download: bool = True) -> Path:
    """确保 ONNX 模型就位，返回模型根目录。已就位则直接返回（幂等）。"""
    version, online_url = _read_upstream_constants()
    roots = _model_roots()
    for root in roots:
        if _models_ready(root, version):
            log.info("[HyperLpr] 模型已就绪: %s", root)
            return root

    if not download:
        raise FileNotFoundError(f"未找到 hyperlpr3 模型（查找路径: {[str(r) for r in roots]}）")

    target = roots[0]
    _download_and_extract(online_url, version, target)
    if not _models_ready(target, version):
        raise RuntimeError(f"模型解压后仍不完整: {target / version / 'onnx'}")
    log.info("[HyperLpr] 模型已下载并解压: %s", target)
    return target


def plate_type_to_color(type_idx: int) -> str:
    """hyperlpr3 车牌类型 idx → 项目底色词（blue/green/yellow/white/black/unknown）。"""
    return _COLOR_BY_TYPE.get(int(type_idx), "unknown")


# ============================================================
# 引擎
# ============================================================

class HyperLprEngine:
    """hyperlpr3 端到端车牌识别引擎（检测 + 识别 + 分类三合一）。

    与 `Detector` / `Recognizer` 协议不同，本引擎一次调用即完成「定位 + 读码」，
    因此直接产出 `PlateResult`，由 `LprPipeline` 作为兜底引擎调用。
    """

    def __init__(self, detect_level: str = "high", logger_level: int = 3, download: bool = True):
        if detect_level not in _CAPACITY:
            raise ValueError(f"detect_level 只能是 {list(_CAPACITY)}，收到 {detect_level!r}")

        ensure_models(download=download)

        import onnxruntime as ort

        from hyperlpr3 import DETECT_LEVEL_HIGH, DETECT_LEVEL_LOW, LicensePlateCatcher
        from hyperlpr3.common.tools_process import get_rotate_crop_image
        from hyperlpr3.common.typedef import DOUBLE

        ort.set_default_logger_severity(logger_level)

        level = DETECT_LEVEL_HIGH if detect_level == "high" else DETECT_LEVEL_LOW
        self.catcher = LicensePlateCatcher(detect_level=level, logger_level=logger_level)
        # 直接持有三个阶段：上游 to_result() 不暴露检测置信度，这里要拿到真实值
        self._detector = self.catcher.pipeline.detector
        self._recognizer = self.catcher.pipeline.recognizer
        self._classifier = self.catcher.pipeline.classifier
        self._get_rotate_crop_image = get_rotate_crop_image
        self._double = DOUBLE
        self.detect_level = detect_level
        log.info("[HyperLpr] 引擎就绪 (detect_level=%s, 输入 %dpx)", detect_level, _CAPACITY[detect_level])

    # --------------------------------------------------------
    def recognize_frame(self, frame: np.ndarray) -> list[PlateResult]:
        """对整帧做端到端识别，返回 PlateResult 列表（检测/识别置信度均为模型真实输出）。"""
        t_frame = time.perf_counter()
        results: list[PlateResult] = []
        for code, rec_conf, det_conf, type_idx, rect, per_plate_ms in self._raw_plates(frame):
            plate_no = code.strip().upper()
            # 不套用 rules 硬过滤：本地规则表未穷举（学/警/港澳/双层牌），硬过滤会误杀。
            # 仅记录校验结论，便于线上观测与后续补规则。
            ok, reason = rules.validate_plate(plate_no)
            if not ok:
                log.info("[HyperLpr] 车牌 %r 未通过本地规则校验(%s)，按模型输出返回", plate_no, reason)

            results.append(
                PlateResult(
                    plate_no=plate_no,
                    plate_color=plate_type_to_color(type_idx),
                    vehicle_type="",  # 车辆类型由 pipeline 用车辆检测器归属
                    det_score=det_conf,
                    rec_score=rec_conf,
                    bbox=[float(v) for v in rect],
                    cost_ms=per_plate_ms,
                )
            )

        results.sort(key=lambda r: r.det_score, reverse=True)
        log.info(
            "[HyperLpr] 帧处理完成: 车牌 %d 个, 总耗时 %.1fms",
            len(results), (time.perf_counter() - t_frame) * 1000,
        )
        return results

    # --------------------------------------------------------
    def _raw_plates(self, frame: np.ndarray) -> list[tuple]:
        """跑检测 → 矫正 → 识别 → 分类，返回 (code, rec_conf, det_conf, type_idx, rect, per_plate_ms)。

        这段逻辑对应上游 `LPRMultiTaskPipeline.run()`，差异是保留下检测置信度与类型判定，
        并且不丢弃未通过其内部 `code_filter` 的结果。
        """
        from hyperlpr3.common.typedef import BLUE, GREEN, UNKNOWN, code_filter

        t_det = time.perf_counter()
        dets = self._detector(frame)
        log.debug("[HyperLpr] 整帧检测耗时 %.1fms, 候选 %d 个",
                  (time.perf_counter() - t_det) * 1000, len(dets))

        out: list[tuple] = []
        for det in dets:
            t_plate = time.perf_counter()
            rect = det[:4].astype(int)
            det_conf = float(det[4])
            landmarks = det[5:13].reshape(4, 2).astype(int)
            layer_num = int(det[13])

            plate_img = self._get_rotate_crop_image(frame, landmarks)
            code, rec_conf = self._recognize_two_layer(plate_img, layer_num)
            if not code:
                continue

            type_idx = code_filter(code)
            if type_idx == UNKNOWN:
                type_idx = self._classify(plate_img, layer_num)
            if type_idx == UNKNOWN:
                type_idx = BLUE if len(code) < 8 else GREEN
            # 单牌耗时 = 矫正 + 识别 + 分类；整帧检测不计入（与自研级联口径一致）
            per_plate_ms = (time.perf_counter() - t_plate) * 1000
            out.append((code, rec_conf, det_conf, type_idx, rect, per_plate_ms))

        return out

    def _recognize_two_layer(self, plate_img: np.ndarray, layer_num: int) -> tuple[str, float]:
        """单层直接识别；双层按 4:6 切上下两行分别识别后拼接。"""
        if layer_num == self._double:
            h = plate_img.shape[0]
            line = int(h * 0.4)
            top_code, top_conf = self._recognizer(plate_img[:line])
            bottom_code, bottom_conf = self._recognizer(plate_img[line:])
            # 强转 Python float：ONNX 输出是 np.float32，round() 后仍是 np.float32，无法 JSON 序列化
            return top_code + bottom_code, float((top_conf + bottom_conf) / 2)
        code, conf = self._recognizer(plate_img)
        return code, float(conf)

    def _classify(self, plate_img: np.ndarray, layer_num: int) -> int:
        """模型分类器判底色，映射回车牌类型 idx。无法判定返回 -1。"""
        from hyperlpr3.common.typedef import (
            BLUE,
            GREEN,
            PLATE_TYPE_BLUE,
            PLATE_TYPE_GREEN,
            PLATE_TYPE_YELLOW,
            UNKNOWN,
            YELLOW_DOUBLE,
            YELLOW_SINGLE,
        )

        idx = int(np.argmax(self._classifier(plate_img)))
        if idx == PLATE_TYPE_BLUE:
            return BLUE
        if idx == PLATE_TYPE_GREEN:
            return GREEN
        if idx == PLATE_TYPE_YELLOW:
            return YELLOW_DOUBLE if layer_num == self._double else YELLOW_SINGLE
        return UNKNOWN

    # --------------------------------------------------------
    def bbox_list(self, frame: np.ndarray) -> list[BBox]:
        """只做检测，返回 BBox 列表（车辆类型归属等场景可用）。"""
        boxes: list[BBox] = []
        for det in self._detector(frame):
            x1, y1, x2, y2 = det[:4]
            boxes.append(BBox(float(x1), float(y1), float(x2), float(y2), float(det[4]), 0))
        return boxes
