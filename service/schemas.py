"""API 请求 / 响应模型（pydantic）。计划书 §10.2 接口定义。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlateItem(BaseModel):
    """单个车牌识别结果。"""

    plate_no: str = Field(..., description="车牌号")
    plate_color: str = Field(..., description="车牌颜色 blue/green/yellow/white/black")
    vehicle_type: str = Field(..., description="车辆类型 car/bus/truck/motorcycle")
    det_score: float = Field(..., description="车牌检测置信度")
    rec_score: float = Field(..., description="识别置信度")
    bbox: list[float] = Field(..., description="车牌框 [x1,y1,x2,y2]")
    cost_ms: float = Field(..., description="单牌耗时(ms)")


class PredictData(BaseModel):
    plates: list[PlateItem] = Field(default_factory=list)


class PredictResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: PredictData = Field(default_factory=PredictData)


class HealthResponse(BaseModel):
    status: str = "ok"
    model_ver: str = "v1.0"
    uptime: float = Field(..., description="服务运行秒数")
    gpu: bool = Field(False, description="是否 GPU 推理")
    engine: str = Field("unknown", description="兜底引擎状态 on/off/unavailable")
    detail: str = Field("", description="降级原因（加载异常时非空）")


class PredictBase64Request(BaseModel):
    image_b64: str = Field(..., description="base64 编码图片")


# ============================================================
# 多图批量
# ============================================================

class BatchItem(BaseModel):
    """批量识别里的单张图片结果（失败时 error 非空、plates 为空）。"""

    name: str = Field("", description="原始文件名")
    plates: list[PlateItem] = Field(default_factory=list)
    error: str = Field("", description="该图的失败原因；成功时为空")
    cost_ms: float = Field(0.0, description="该图耗时(ms)")


class BatchPredictData(BaseModel):
    items: list[BatchItem] = Field(default_factory=list)
    count: int = Field(0, description="收到的图片数")
    ok_count: int = Field(0, description="识别成功（含空结果）的图片数")
    total_plates: int = Field(0, description="识别出的车牌总数")
    elapsed_ms: float = Field(0.0, description="总耗时(ms)")


class BatchPredictResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: BatchPredictData = Field(default_factory=BatchPredictData)


# ============================================================
# 视频异步任务
# ============================================================

class VideoJobStartData(BaseModel):
    """提交视频后立即返回：只给任务号，处理在后台线程里跑。"""

    job_id: str
    status: str = Field(..., description="queued / running")
    source: str = Field("", description="原始文件名")
    window_s: float = Field(3.0, description="去重时间窗(秒)")
    max_frames: int = Field(0, description="处理帧数上限，0=用服务端默认上限")
    poll_url: str = Field("", description="轮询进度的相对地址")


class VideoProgress(BaseModel):
    frames: int = Field(0, description="已处理帧数")
    total_frames: int = Field(0, description="容器声称的总帧数，0=未知")
    percent: float = Field(0.0, description="进度百分比（total 未知时为 0）")
    det_frames: int = Field(0, description="含车牌的帧数")
    elapsed_s: float = Field(0.0, description="已耗时(秒)")
    eta_s: float | None = Field(None, description="预计剩余(秒)，样本不足时为 null")


class VideoEventItem(BaseModel):
    """一个车牌出现事件（去重后）。"""

    plate_no: str = ""
    plate_color: str = ""
    vehicle_type: str = ""
    det_score: float = 0.0
    rec_score: float = 0.0
    bbox: list[float] = Field(default_factory=list)
    frame_idx: int = 0
    t_sec: float = 0.0
    timestamp: str = ""
    hits: int = 1
    shot: str = Field("", description="最佳帧小图文件名（相对 shots 目录）")


class VideoUrls(BaseModel):
    """产物地址（均为相对路径，前端自行拼 API 前缀）。"""

    annotated: str = Field("", description="标注视频")
    csv: str = Field("", description="事件 CSV")
    jsonl: str = Field("", description="事件 JSONL")
    shots: str = Field("", description="最佳帧小图目录")


class VideoJobStatusData(BaseModel):
    job_id: str
    status: str = Field(..., description="queued / running / done / failed")
    source: str = ""
    window_s: float = 3.0
    max_frames: int = 0
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    progress: VideoProgress = Field(default_factory=VideoProgress)
    stats: dict = Field(default_factory=dict, description="完成后的统计汇总")
    events: list[VideoEventItem] = Field(default_factory=list, description="完成后的车牌事件列表")
    urls: VideoUrls = Field(default_factory=VideoUrls)
    error: str = Field("", description="失败原因；成功时为空")


class VideoJobStatusResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: VideoJobStatusData


class VideoJobStartResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: VideoJobStartData


# ============================================================
# 实时流会话
# ============================================================

class StreamStartData(BaseModel):
    """启动实时会话后立即返回：前端把 `mjpeg_url` 塞进 `<img src>` 就能看到画面。"""

    session_id: str
    status: str = Field(..., description="starting / running / failed")
    source: str = Field("", description="来源展示名（文件名 / 摄像头索引 / RTSP 地址）")
    mjpeg_url: str = Field("", description="MJPEG 推流地址（相对路径）")
    status_url: str = Field("", description="状态轮询地址（相对路径）")
    media_base: str = Field("", description="事件截图目录（相对路径）")
    interval_ms: int = Field(0, description="识别间隔(ms)，0=每帧尽力")
    max_side: int = Field(640, description="画面长边上限（越宽越慢）")
    level: str = Field("low", description="检测档位 low(320px)/high(640px)")


class StreamStartResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: StreamStartData


class StreamStats(BaseModel):
    frames: int = Field(0, description="已推送给客户端的帧数")
    read_frames: int = Field(0, description="已从源读到的帧数")
    skipped: int = Field(0, description="为追实时而丢弃的帧数（直播源）")
    recog_runs: int = Field(0, description="实际执行的识别次数")
    recog_dropped: int = Field(0, description="因识别算力不足被丢弃的帧数（队列满即丢，只认最新帧）")
    avg_recog_ms: float = Field(0.0, description="单次识别平均耗时(ms)")
    events: int = Field(0, description="去重后的车牌事件数")
    total_hits: int = Field(0, description="命中帧数合计")
    elapsed_s: float = Field(0.0, description="已运行秒数")
    push_fps: float = Field(0.0, description="实际推送帧率")
    recog_fps: float = Field(0.0, description="实际识别频率（次/秒）")
    source_fps: float | None = Field(None, description="源帧率，未知为 null")
    source_frames: int | None = Field(None, description="源总帧数，直播源为 null")
    live: bool = Field(False, description="是否直播源（摄像头/RTSP）")
    interval_ms: int = 0
    max_side: int = 640
    jpeg_quality: int = 75
    warmup_ms: float = Field(0.0, description="启动阶段的预热耗时(ms)：把它从首次识别里挪走了")
    first_frame_ms: float = Field(0.0, description="首帧到达耗时(ms)：设备慢/被占用时一眼可见")
    backend: str = Field("", description="采集后端名（摄像头才有，如 DSHOW）")


class CameraDevice(BaseModel):
    """一个摄像头索引的探测结果。"""

    index: int
    ok: bool = Field(False, description="能否打开**并读到一帧**（只打开不算可用）")
    backend: str = ""
    first_frame_ms: float = 0.0
    width: int = 0
    height: int = 0
    error: str = ""


class CameraDevicesData(BaseModel):
    devices: list[CameraDevice] = Field(default_factory=list)
    available: list[int] = Field(default_factory=list, description="可用索引")
    note: str = Field("", description="给用户看的建议（例如没有可用摄像头时）")


class CameraDevicesResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: CameraDevicesData


class StreamStatusData(BaseModel):
    session_id: str
    status: str = Field(..., description="starting / running / stopped / failed")
    source: str = ""
    source_title: str = Field("", description="网页链接解析出的视频标题（非网页来源为空）")
    error: str = Field("", description="失败原因；正常时为空")
    stats: StreamStats = Field(default_factory=StreamStats)
    events: list[VideoEventItem] = Field(default_factory=list, description="实时累积的车牌事件")
    media_base: str = Field("", description="截图目录相对地址")


class StreamStatusResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: StreamStatusData


# ============================================================
# 浏览器摄像头（帧由前端推上来，服务端只识别）
# ============================================================

class CameraSessionStartData(BaseModel):
    """创建浏览器摄像头会话后返回：前端据此开始推帧并轮询状态。"""

    session_id: str
    status: str = Field(..., description="starting / running / stopped / failed")
    frame_url: str = Field("", description="推帧地址（相对路径，POST multipart，字段名 file）")
    status_url: str = Field("", description="状态轮询地址（相对路径）")
    media_base: str = Field("", description="事件截图目录（相对路径）")
    interval_ms: int = Field(200, description="识别间隔(ms)：推帧不必快于此值")
    max_side: int = Field(640, description="画面长边上限（服务端会再降采样一次）")
    level: str = Field("low", description="检测档位 low/high")
    max_frame_bytes: int = Field(..., description="单帧上传大小上限(字节)")
    info: str = Field("", description="给用户看的说明（隐私/授权相关）")


class CameraSessionStartResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: CameraSessionStartData


class CameraBoxItem(BaseModel):
    """画面上的一个框：kind 决定前端怎么画。

    - `plate`     车牌（含识别结果）
    - `vehicle`   车辆 / `person` 行人（细线，说明"检测到了什么"）
    - `candidate` 疑似车牌但没过阈值（灰线 + 原因，回答"为什么没识别出"）
    """

    kind: str = Field(..., description="plate / vehicle / person / candidate")
    label: str = Field("", description="展示文案（车牌号 / 类别 / 未通过原因）")
    score: float = Field(0.0, description="置信度")
    bbox: list[float] = Field(default_factory=list, description="[x1,y1,x2,y2]，帧像素坐标")
    plate_color: str = Field("", description="仅车牌：blue/green/yellow/white/black")
    rec_score: float = Field(0.0, description="仅车牌：识别置信度")


class CameraFrameData(BaseModel):
    """一帧的识别结果。坐标基于 `frame_size`，前端换算到显示尺寸后画框。"""

    session_id: str
    status: str = ""
    reused: bool = Field(False, description="本帧沿用了上次结果（被节流/推理占用），非错误")
    boxes: list[CameraBoxItem] = Field(default_factory=list, description="本次要画的全部框")
    frame_size: list[int] = Field(default_factory=list, description="[宽, 高]，box 坐标所在空间")
    stats: StreamStats = Field(default_factory=StreamStats)
    events: list[VideoEventItem] = Field(default_factory=list, description="累积的车牌事件（去重后）")
    media_base: str = Field("", description="截图目录相对地址")


class CameraFrameResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: CameraFrameData
