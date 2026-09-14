"""FastAPI 服务。计划书 §10：/health + /predict + /predict_base64 + 上传识别页。

启动：uvicorn service.app:app --host 0.0.0.0 --port 8000
页面：http://127.0.0.1:8000/          （service/static/index.html）
接口文档：http://127.0.0.1:8000/docs

接口一览
--------
    GET  /health                  探活（含兜底引擎状态）
    POST /predict                 单图识别
    POST /predict_base64          base64 单图识别
    POST /predict_batch           多图批量识别（一张失败不影响其余）
    POST /predict_video           提交视频识别任务（异步，立即返回 job_id）
    GET  /predict_video/{job_id}  查询任务进度 / 结果
    POST /stream                  启动实时识别会话（摄像头 / RTSP / 本地视频）
    GET  /stream/{id}.mjpg        MJPEG 推流（边播边识别，`<img>` 直接显示）
    GET  /stream/{id}             实时会话状态 / 已识别车牌
    DELETE /stream/{id}           停止会话（?purge=true 连产物一起删）
    GET  /                       上传识别页
    GET  /media/...               视频任务产物（标注视频 / CSV / 车牌小图）
    GET  /stream-media/...        实时会话产物（车牌小图）

三条长耗时路径的取舍
-------------------
| 场景 | 接口 | 为什么这么设计 |
|---|---|---|
| 一段视频，要完整结果 | `POST /predict_video` | 异步任务：处理要分钟级，同步 HTTP 必超时，轮询才有真进度 |
| 视频/摄像头，要边播边看 | `POST /stream` | MJPEG 推流：浏览器不能直连 RTSP，只有服务端解码再推这一条路 |
| 单图/多图 | `/predict`、`/predict_batch` | 秒级完成，同步返回最省事 |
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

# 允许以 uvicorn service.app:app 方式直接启动
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from service.jobs import VideoJobManager, _remove_tree  # noqa: E402
from service.schemas import (  # noqa: E402
    BatchItem,
    BatchPredictData,
    BatchPredictResponse,
    CameraDevice,
    CameraDevicesData,
    CameraDevicesResponse,
    HealthResponse,
    PlateItem,
    PredictBase64Request,
    PredictData,
    PredictResponse,
    StreamStartData,
    StreamStartResponse,
    StreamStatusData,
    StreamStatusResponse,
    VideoJobStartData,
    VideoJobStartResponse,
    VideoJobStatusData,
    VideoJobStatusResponse,
)
from service.streams import StreamManager, TooManyStreams  # noqa: E402
from src.common.logger import setup_logger  # noqa: E402
from src.io.reader import probe_cameras  # noqa: E402

setup_logger("service")
log = logging.getLogger("service")

START_TIME = time.time()
MODEL_VER = "v1.0"
BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = BASE_DIR / "runs"
JOBS_DIR = RUNS_DIR / "jobs"
STREAMS_DIR = RUNS_DIR / "streams"

# 上传限制
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_BATCH_FILES = 20
UPLOAD_CHUNK = 1 << 20  # 1MB：边收边落盘，避免整段视频进内存

VIDEO_SUFFIXES = {
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".mpg", ".mpeg", ".ts",
}

# MJPEG 分隔串（multipart/x-mixed-replace）
MJPEG_BOUNDARY = "plateframeboundary"

# 摄像头探测：探测几个索引、整体限时多久（驱动卡住时不能把请求也拖住）
CAMERA_PROBE_COUNT = 4
CAMERA_PROBE_TIMEOUT_S = 12.0

# 全局 pipeline（启动时加载，供单图 / 批量接口使用）
_pipeline = None
_load_error: str | None = None


def _build_pipeline():
    """按 configs/lprnet.yaml 构建流水线（与 CLI 共用同一份构建逻辑，避免两处配置漂移）。"""
    from src.pipeline.lpr_pipeline import build_pipeline

    return build_pipeline()


def _build_video_pipeline():
    """视频任务专用的流水线实例。

    **刻意不复用 `_pipeline`**：视频任务耗时以分钟计，若与单图接口共用同一份模型实例，
    一是要给它加锁（单图请求会被长任务拖住），二是 ultralytics / 单例 predictor 在
    跨线程并发下并不安全。多花几十 MB 内存换"两条链路互不阻塞"，值得。
    实例由 `VideoJobManager` 缓存复用（单 worker 串行，仍是单线程访问）。
    """
    from src.pipeline.lpr_pipeline import build_pipeline

    return build_pipeline()


def _build_stream_pipeline(level: str | None = None,
                           min_det_score: float | None = None,
                           min_rec_score: float | None = None):
    """实时会话专用流水线：允许按会话覆盖检测档位与识别阈值。

    阈值覆盖是给"玩具车 / 小车牌 / 远距离车牌"用的：默认 0.5/0.6 会把它们直接丢掉。
    同样每路会话独立实例——实时流是长期驻留线程，跨线程共享模型实例不安全。
    """
    from src.pipeline.lpr_pipeline import build_pipeline

    return build_pipeline(engine_level=level,
                          min_det_score=min_det_score, min_rec_score=min_rec_score)


_video_manager = VideoJobManager(JOBS_DIR, pipeline_factory=_build_video_pipeline)
_stream_manager = StreamManager(STREAMS_DIR, pipeline_factory=_build_stream_pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _load_error
    try:
        _pipeline = _build_pipeline()
        log.info("[Service] 模型加载完成")
    except Exception as exc:  # 权重缺失/下载失败时降级，服务仍可响应 /health
        _load_error = str(exc)
        log.error("[Service] 模型加载失败（降级运行）: %s", exc)
    # 启动时顺手清掉过期产物（服务被反复重启也不会把磁盘堆满）
    for name, purge in (("视频任务", _video_manager.purge_expired), ("实时会话", _stream_manager.purge)):
        try:
            purge()
        except Exception as exc:  # 清理失败不该影响启动
            log.warning("[Service] 清理过期%s失败: %s", name, exc)
    yield
    _video_manager.shutdown(wait=False)
    _stream_manager.stop_all(timeout=1.0)


app = FastAPI(
    title="车牌与机动车识别系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 允许用 file:// 或其它端口打开页面时直接调用接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态资源（示例图等）；上传识别页单独挂在根路径
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 产物目录：目录名是服务端生成的 hex 编号，无路径穿越风险
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STREAMS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=JOBS_DIR), name="media")
app.mount("/stream-media", StaticFiles(directory=STREAMS_DIR), name="stream_media")


def _html_page(filename: str) -> FileResponse:
    """返回页面文件，并显式禁用缓存。

    为什么必须 no-store：FileResponse 默认只带 ETag/Last-Modified，**没有 Cache-Control**，
    浏览器于是按"启发式缓存"规则（约 last-modified 之后时长的 10%）直接吃本地副本、
    不做校验——实测改了首页（/ 从上传页换成实时页）之后，用户浏览器仍显示旧页面，
    而服务器返回的已是新页面。页面是入口，必须每次都拿最新的。
    同时放开 HEAD：健康探测/监控常用 HEAD，只注册 GET 会返回 405。
    """
    page = STATIC_DIR / filename
    if not page.is_file():
        raise HTTPException(status_code=404, detail=f"页面缺失: service/static/{filename}")
    return FileResponse(page, headers={"Cache-Control": "no-store, must-revalidate"})


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def live_page() -> FileResponse:
    """首页 = 实时流监控页（摄像头 / RTSP / 网页链接，边播边识别）。"""
    return _html_page("live.html")


@app.api_route("/live", methods=["GET", "HEAD"], include_in_schema=False)
def live_page_alias() -> FileResponse:
    """兼容旧链接：/live 与首页同一个页面。"""
    return _html_page("live.html")


@app.api_route("/upload", methods=["GET", "HEAD"], include_in_schema=False)
def upload_page() -> FileResponse:
    """上传识别页（图片单张/批量 / 视频异步任务）。"""
    return _html_page("index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查：报告模型版本、运行时长、识别路径与降级原因。"""
    if _pipeline is None:
        engine_state = "unavailable"
    else:
        engine_state = "on" if _pipeline.engine is not None else "off"
    return HealthResponse(
        status="ok",
        model_ver=MODEL_VER,
        uptime=round(time.time() - START_TIME, 1),
        gpu=False,
        engine=engine_state,
        detail=_load_error or "",
    )


# ============================================================
# 单图 / 批量
# ============================================================

def _bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片")
    return img


def _require_pipeline():
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"模型未就绪: {_load_error or '加载中'}",
        )
    return _pipeline


def _run_predict(img: np.ndarray) -> PredictResponse:
    results = _require_pipeline().run(img)
    plates = [PlateItem(**r.to_dict()) for r in results]
    return PredictResponse(data=PredictData(plates=plates))


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """上传图片文件，返回车牌识别结果 JSON。"""
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"图片超过 {MAX_IMAGE_BYTES // 1048576} MB")
    img = _bytes_to_bgr(data)
    return _run_predict(img)


@app.post("/predict_base64", response_model=PredictResponse)
async def predict_base64(req: PredictBase64Request) -> PredictResponse:
    """base64 图片识别。"""
    try:
        data = base64.b64decode(req.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="base64 解码失败") from exc
    img = _bytes_to_bgr(data)
    return _run_predict(img)


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(files: list[UploadFile] = File(...)) -> BatchPredictResponse:
    """多图批量识别：单张失败只影响它自己，整批仍然返回 200。

    逐张串行推理（CPU 上并行只会互相抢核）；每张的耗时单独上报，便于定位慢图。
    """
    _require_pipeline()
    if not files:
        raise HTTPException(status_code=400, detail="没有收到文件")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"一次最多 {MAX_BATCH_FILES} 张")

    items: list[BatchItem] = []
    t0 = time.perf_counter()
    for f in files:
        name = f.filename or "image"
        try:
            data = await f.read()
            if len(data) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail=f"超过 {MAX_IMAGE_BYTES // 1048576} MB")
            img = _bytes_to_bgr(data)
        except HTTPException as exc:
            items.append(BatchItem(name=name, error=str(exc.detail)))
            continue
        t1 = time.perf_counter()
        try:
            results = _pipeline.run(img)
        except Exception as exc:  # 单张推理异常不该拖垮整批
            log.exception("[Batch] 识别失败: %s", name)
            items.append(BatchItem(name=name, error=f"识别失败: {exc}"))
            continue
        items.append(BatchItem(
            name=name,
            plates=[PlateItem(**r.to_dict()) for r in results],
            cost_ms=round((time.perf_counter() - t1) * 1000, 1),
        ))

    return BatchPredictResponse(data=BatchPredictData(
        items=items,
        count=len(items),
        ok_count=sum(1 for it in items if not it.error),
        total_plates=sum(len(it.plates) for it in items),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    ))


# ============================================================
# 视频（异步任务）
# ============================================================

def _clamp_score(value: float, default: float) -> float:
    """把识别阈值收敛到 [0.05, 0.95]；解析失败用默认值（防止前端传 0 把结果刷屏）。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.05, min(0.95, value))


async def _save_upload(upload: UploadFile, dest: Path, limit: int) -> int:
    """边收边落盘，返回字节数；超限抛 413 并删掉半成品。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                fh.close()
                try:
                    dest.unlink(missing_ok=True)
                except OSError as exc:  # 受限环境可能删不掉，不该因此把 413 变成 500
                    log.warning("[Service] 清理半成品失败（不影响响应）: %s", exc)
                raise HTTPException(status_code=413,
                                    detail=f"文件超过 {limit // 1048576} MB，请先压缩或截短")
            fh.write(chunk)
    return size


@app.post("/predict_video", response_model=VideoJobStartResponse)
async def predict_video(
    file: UploadFile = File(...),
    window_s: float = Form(3.0),
    max_frames: int = Form(0),
) -> VideoJobStartResponse:
    """上传视频，提交识别任务，立即返回任务号（用 GET 轮询进度）。"""
    _require_pipeline()  # 模型没起来就别让用户白传 200MB

    name = file.filename or "upload.mp4"
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式 {suffix}；支持 {', '.join(sorted(VIDEO_SUFFIXES))}",
        )
    if not 0 <= window_s <= 30:
        raise HTTPException(status_code=400, detail="window_s 需在 0~30 秒之间")

    job = _video_manager.create(name, window_s=window_s, max_frames=max_frames, suffix=suffix or ".mp4")
    try:
        size = await _save_upload(file, job.input_path, MAX_VIDEO_BYTES)
    except HTTPException:
        _video_manager.discard(job.job_id)   # 别留一个永远不会跑的空任务和空目录
        raise
    if size == 0:
        _video_manager.discard(job.job_id)
        raise HTTPException(status_code=400, detail="收到 0 字节文件")

    _video_manager.submit(job)
    log.info("[Service] 视频任务已提交: %s (%s, %.1f MB, window=%.1fs, max_frames=%d)",
             job.job_id, name, size / 1048576, window_s, job.max_frames)
    return VideoJobStartResponse(data=VideoJobStartData(
        job_id=job.job_id,
        status=job.status,
        source=name,
        window_s=job.window_s,
        max_frames=job.max_frames,
        poll_url=f"/predict_video/{job.job_id}",
    ))


@app.get("/predict_video/{job_id}", response_model=VideoJobStatusResponse)
async def predict_video_status(job_id: str) -> VideoJobStatusResponse:
    """查询视频任务进度；done 时附带事件列表与产物地址。"""
    snap = _video_manager.snapshot(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="任务不存在（可能已被过期清理）")
    return VideoJobStatusResponse(data=VideoJobStatusData(**snap))


@app.delete("/predict_video/{job_id}")
async def predict_video_cancel(job_id: str) -> dict:
    """丢弃任务产物（排队中/已完成都可）。

    运行中的任务不会被打断（CPU 推理无法安全抢占），但产物即刻从任务列表移除。
    受限环境（装了 safe-delete 之类删除钩子）可能删不掉磁盘文件，此时 `purged=false`
    会如实返回并附带 `note`，绝不假装删干净了。
    """
    info = _video_manager.discard(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    data = {"discarded": True, **info}
    if not info["purged"]:
        data["note"] = ("当前环境拦截了批量删除，产物仍保留在磁盘；"
                        "可执行 python tools/clean_jobs.py --all 手动清理")
    return {"code": 0, "msg": "success", "data": data}


# ============================================================
# 实时流（MJPEG 推流）
# ============================================================

@app.post("/stream", response_model=StreamStartResponse)
async def stream_start(
    file: UploadFile | None = File(None),
    source: str = Form(""),
    interval_ms: int = Form(200),
    level: str = Form("low"),
    max_side: int = Form(640),
    jpeg_quality: int = Form(75),
    min_det: float = Form(0.5),
    min_rec: float = Form(0.6),
    show_objects: bool = Form(True),
) -> StreamStartResponse:
    """启动一路实时识别会话：本机摄像头 / RTSP-RTMP 地址 / 上传的本地视频。

    立即返回 `mjpeg_url`，前端把它塞进 `<img src>` 就能看到"边播边识别"的画面；
    实时统计与已识别车牌用 `GET /stream/{id}` 轮询。

    `min_det` / `min_rec`：识别阈值（可按会话调低）。玩具车、小车牌、远距离车牌经常
    低于默认 0.5/0.6 而被直接过滤——调低后它们才有机会被读出来。
    `show_objects`：是否在画面上标注车辆/行人与"疑似车牌"（未过阈值的候选）。
    """
    _require_pipeline()  # 模型没起来就别让用户白等
    level = level if level in ("low", "high") else "low"
    min_det = _clamp_score(min_det, 0.5)
    min_rec = _clamp_score(min_rec, 0.6)

    session_id = uuid4().hex[:16]
    out_dir = STREAMS_DIR / session_id
    display_name = source.strip()

    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix and suffix not in VIDEO_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的视频格式 {suffix}；支持 {', '.join(sorted(VIDEO_SUFFIXES))}",
            )
        input_path = out_dir / f"input{suffix or '.mp4'}"
        try:
            size = await _save_upload(file, input_path, MAX_VIDEO_BYTES)
        except HTTPException:
            _remove_tree(out_dir)
            raise
        if size == 0:
            _remove_tree(out_dir)
            raise HTTPException(status_code=400, detail="收到 0 字节文件")
        src: str | int = str(input_path)
        display_name = file.filename
    elif display_name:
        if display_name.isdigit():
            src = int(display_name)
            display_name = f"摄像头 {display_name}"
        elif display_name.lower().startswith(
                ("rtsp://", "rtsps://", "rtmp://", "udp://", "tcp://", "http://", "https://")):
            # http(s) 允许：远端视频文件也走这条路（cv2 经 FFmpeg 打开）。
            # 注意这意味着服务端会去访问调用方给的地址——**本工具定位是内网/本机调试**，
            # 需要对外暴露时请在网关层做白名单，不要直接暴露到公网。
            src = display_name
        else:
            raise HTTPException(
                status_code=400,
                detail="source 只支持摄像头索引（如 0）或 rtsp:// / rtmp:// / http(s):// 等地址",
            )
    else:
        raise HTTPException(status_code=400, detail="请提供 file（本地视频）或 source（摄像头/流地址）")

    try:
        # 构建流水线要几秒，放线程池，别把事件循环卡住
        session = await run_in_threadpool(
            _stream_manager.create, session_id, src,
            out_dir=out_dir, name=display_name, interval_ms=interval_ms,
            level=level, max_side=max_side, jpeg_quality=jpeg_quality,
            min_det_score=min_det, min_rec_score=min_rec, show_objects=show_objects,
        )
    except TooManyStreams as exc:
        _remove_tree(out_dir)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("[Service] 启动实时会话失败")
        _remove_tree(out_dir)
        raise HTTPException(status_code=500, detail=f"启动实时会话失败: {exc}") from exc

    log.info("[Service] 实时会话已启动: %s source=%s interval=%dms level=%s size=%d",
             session_id, display_name, session.interval_s * 1000, level, session.max_side)
    return StreamStartResponse(data=StreamStartData(
        session_id=session_id,
        status=session.status,
        source=display_name,
        mjpeg_url=f"/stream/{session_id}.mjpg",
        status_url=f"/stream/{session_id}",
        media_base=f"/stream-media/{session_id}/shots" if session.shots is not None else "",
        interval_ms=int(round(session.interval_s * 1000)),
        max_side=session.max_side,
        level=level,
    ))


@app.get("/stream/devices", response_model=CameraDevicesResponse)
async def stream_devices(count: int = CAMERA_PROBE_COUNT) -> CameraDevicesResponse:
    """探测本机可用的摄像头（**能打开且能读到一帧**才算可用）。

    页面靠它决定默认来源：没有可用摄像头时，不该让用户一上来就点到一个黑屏设备。
    探测可能被驱动卡住，所以放线程池 + 限时；超时就如实说明探测未完成。
    （注意本路由必须注册在 `/stream/{session_id}` **之前**，否则 "devices" 会被当成会话号。）
    """
    n = max(1, min(int(count or CAMERA_PROBE_COUNT), 8))
    try:
        raw = await asyncio.wait_for(
            run_in_threadpool(probe_cameras, n), timeout=CAMERA_PROBE_TIMEOUT_S)
    except asyncio.TimeoutError:
        log.warning("[Service] 摄像头探测超时（%.0fs）", CAMERA_PROBE_TIMEOUT_S)
        return CameraDevicesResponse(data=CameraDevicesData(
            devices=[], available=[],
            note=f"摄像头探测超过 {CAMERA_PROBE_TIMEOUT_S:.0f}s 未返回（设备可能被占用或驱动卡住）"))

    devices = [CameraDevice(**d) for d in raw]
    available = [d.index for d in devices if d.ok]
    if available:
        note = f"可用摄像头：{', '.join(str(i) for i in available)}"
    else:
        note = "未检测到可用摄像头（能打开但读不到帧的不算）。可改用「本地视频」，或点「示例视频」直接看效果。"
    log.info("[Service] 摄像头探测: 可用 %s", available or "无")
    return CameraDevicesResponse(data=CameraDevicesData(
        devices=devices, available=available, note=note))


async def _mjpeg_frames(session):
    """把会话的最新帧按 MJPEG 格式吐给客户端。

    `wait_frame` 会阻塞（最多 5s），因此丢到线程池执行，避免卡住事件循环。
    客户端断开时 Starlette 会取消这个生成器；会话本身继续跑，由空闲超时负责收尾。
    """
    seq = -1
    while True:
        seq, jpeg = await run_in_threadpool(session.wait_frame, seq, 5.0)
        if jpeg is None:
            if session.is_finished:
                break
            continue      # 只是暂时没有新帧（识别慢于 display_fps），继续等
        header = (f"--{MJPEG_BOUNDARY}\r\n"
                  f"Content-Type: image/jpeg\r\n"
                  f"Content-Length: {len(jpeg)}\r\n\r\n").encode("ascii")
        yield header + jpeg + b"\r\n"


@app.get("/stream/{session_id}.mjpg", include_in_schema=False)
async def stream_mjpeg(session_id: str) -> StreamingResponse:
    """MJPEG 推流：浏览器 `<img src="/stream/<id>.mjpg">` 即可实时播放带框画面。"""
    session = _stream_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return StreamingResponse(
        _mjpeg_frames(session),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@app.get("/stream/{session_id}", response_model=StreamStatusResponse)
async def stream_status(session_id: str) -> StreamStatusResponse:
    """实时会话状态：运行中看统计，结束后仍可读事件列表与截图。"""
    snap = _stream_manager.snapshot(session_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="会话不存在（可能已被清理）")
    return StreamStatusResponse(data=StreamStatusData(**snap))


@app.delete("/stream/{session_id}")
async def stream_stop(session_id: str, purge: bool = False) -> dict:
    """停止实时会话。默认**保留记录与截图**（便于回看识别结果）；`?purge=true` 连产物一起删。"""
    if purge:
        info = _stream_manager.remove(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        data = {"discarded": True, **info}
        if not info["purged"]:
            data["note"] = "当前环境拦截了删除，产物仍在磁盘；可执行 python tools/clean_jobs.py --all"
        return {"code": 0, "msg": "success", "data": data}
    if not _stream_manager.stop(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"code": 0, "msg": "success", "data": {"session_id": session_id, "stopped": True}}
