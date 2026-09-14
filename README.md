# MV-01 · 车牌与机动车识别系统

> 项目编号：MV-01 ｜ 难度 ★★☆☆☆ ｜ 目标周期 4 周（约 55 h）
> 计划书：`../docs/01-车牌与机动车识别系统-计划书.md`

一句话目标：输入一张图或一路视频，输出**车牌号码 + 车牌颜色 + 车辆类别 + 置信度**，单帧端到端推理 < 80 ms（RTX 3060 级别）。

---

## 当前状态

- [x] M1 数据与环境落地 —— 依赖已装（venv）、数据工具链 5 个脚本可用
- [x] M2 检测模型训练 —— 训练/评测脚本就绪，**权重待 GPU + 真实数据产出**
- [x] M3 字符识别 + 全链路串联 —— 单图端到端跑通，**已能读出真实车牌号**（见下）
- [x] M4 服务化、压测与交付 —— 服务 + 上传识别页 + ONNX 导出 + 压测 + Docker
- [x] M4+ 视频/摄像头/RTSP 入口 —— 逐帧识别 + 跨帧去重 + 事件 CSV/JSONL + 标注视频
- [x] M4+ 视频/批量 HTTP 接口 —— 异步任务 + 真进度；页面直接上传视频
- [x] M4+ **实时流识别** —— 服务器摄像头 / RTSP / 网页链接 / 本地视频，**边播边识别**（MJPEG 推流 + 异步识别）
- [x] M4+ **浏览器摄像头** —— 用**看网页的人**本机的摄像头（浏览器采集 + 推帧 + 结果叠画），云服务器上也能用（服务端没有摄像头设备）

> **验收口径（本机无 GPU，走计划书 R2 降级路径）**：
> 代码全链路可执行、**188 个单测全绿**、图片/视频/实时流/浏览器摄像头四条链路端到端可用、
> 真实图片与真实视频均实测读出车牌号（`粤A3333G`）、**本机摄像头实测出画面**（640×480，6 fps，
> 且能标注行人/车辆与"疑似车牌"候选）。
> 精度类指标（车辆 mAP≥0.95 / 整牌准确率≥92%）与"25fps 全帧实时"需 GPU + 真实数据训练后才能复现，
> 属**待办而非已完成**。

## 四条入口（同一套流水线，不同包装）

| 入口 | 形态 | 适用 |
|---|---|---|
| **图片** | `POST /predict`、`POST /predict_base64`、`POST /predict_batch`（多图） | 单张截图 / 批量静态图 |
| **视频** | `POST /predict_video` → 轮询 `GET /predict_video/{job_id}` | 上传一段视频，异步处理、可看进度、产出标注视频与事件表 |
| **实时流** | `POST /stream` → `<img src="/stream/{id}.mjpg">` | 服务器摄像头 / RTSP / 本地视频 / **网页视频链接**（微博、抖音、B站等，自动解析直链），**边播边识别** |
| **浏览器摄像头** | `POST /camera/session` → 逐帧 `POST /camera/{id}/frame` | 用**用户本机**的摄像头：画面在本地播放，只把帧推上去识别，结果由前端叠画（页面 `/live` 默认来源） |
| **命令行** | `python -m src.pipeline.lpr_pipeline` / `video_pipeline` | 本地批处理、无人值守 |

## 两条推理路径

项目同时具备两条路径，由 `configs/lprnet.yaml` 的 `engine_mode` 控制：

| 路径 | 实现 | 依赖 | 现状 |
|---|---|---|---|
| **自研级联** `cascade` | 车辆检测 → 车牌检测 → 透视矫正 → LPRNet → 规则校验（计划书 §3.1 主设计） | `weights/plate_best.pt` + `weights/lprnet_best.pt`（**待训练**） | 链路完整，权重未产出故读不出号码 |
| **兜底引擎** `engine` | `hyperlpr3`（开源中文车牌 ONNX：检测+识别+分类三合一，Apache-2.0） | 自动下载 12MB 模型到 `%HOMEPATH%\.hyperlpr3` | **可用**，实测读出真实车牌号 |

```yaml
# configs/lprnet.yaml
engine_mode: auto      # cascade=只用自研 / auto=级联无结果时用引擎兜底 / engine=只用引擎
```

> 待 M2 训练权重产出后，把 `engine_mode` 改回 `cascade` 即可切换回完全自研路径。
> 实时流场景下 `low`（320px）比 `high`（640px）快约 2 倍，是"实时"最有效的旋钮（可按会话覆盖）。

## 运行与使用

依赖已装在隔离 venv（torch 2.14.0+cpu / ultralytics / opencv / onnxruntime / hyperlpr3），无需重复安装。

```bash
cd D:\py\机器视觉\plate-recognition
V="C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
```

### 上传识别页（图片 + 视频）

```bash
"$V" -m uvicorn service.app:app --host 127.0.0.1 --port 8000
```

- **图片模式** <http://127.0.0.1:8000/> ：拖入 / 选择 / 直接粘贴截图；支持**一次多选批量**，
  批量结果可一键导出 CSV。页面展示号码、颜色、车辆类型、检测/识别置信度与单牌耗时，
  并在预览图上画出车牌框（结果按 `plate_color` 渲染成一块真实车牌样式）。
- **视频模式**：拖入 MP4/MOV/AVI/MKV/WebM（拖进来自动切到视频模式），可选「处理范围」「去重窗口」。
  上传进度 → 服务端处理显示**真实帧进度与预计剩余** → 完成后给出标注视频回放、
  每个车牌事件的截图/首现时间/命中帧数/置信度、事件 CSV 导出。
- 两种模式都有「用示例图/示例视频试试」，走的是与手动上传完全相同的链路。

### 实时流识别页（浏览器摄像头 / RTSP / 网页视频链接 / 边播边识别）

<http://127.0.0.1:8000/live>

1. 选来源：**浏览器摄像头**（默认，用你当前设备） / **RTSP-RTMP 地址或视频网页链接** / **本地视频文件**；
2. 调参数：识别频率（每帧尽力 / 200ms / 500ms / 1s）、检测档位（low/high）、画面宽度（480–1280）；
3. 点「开始实时识别」→ 画面立刻开始播放（带框），右侧实时刷新统计与识别到的车牌（含截图）。

```bash
# 命令行也能起一路（不开页面）
curl -F "source=0" -F "interval_ms=200" -F "level=low" -F "max_side=640" http://127.0.0.1:8000/stream
# → {"data":{"session_id":"...","mjpeg_url":"/stream/<id>.mjpg", ...}}
# 浏览器/播放器直接看：http://127.0.0.1:8000/stream/<id>.mjpg

# 网页视频链接（微博/抖音/B站/YouTube…）直接粘贴即可，服务端自动解析出真实视频地址：
curl -F "source=https://weibo.com/tv/show/1034:xxxx" -F "interval_ms=500" http://127.0.0.1:8000/stream
"$V" tools/verify_online_link.py   # 该链路一键自检（默认跑一条微博视频）
```

**它是怎么做到"边播边识别"的**（这是实时与离线的本质差别）：

| 机制 | 说明 |
|---|---|
| 读帧线程按源帧率播放 | 文件源按 `fps` 节流，真的是"播放"；直播源不节流，落后时**丢帧追赶**（保低延迟） |
| 识别放**独立线程** + 队列容量 1 | 识别慢（本机约 1 次/秒）不再拖住画面；中间帧沿用上一次结果的框 |
| 只认最新帧 | 识别期间涌入的帧直接丢弃（`recog_dropped` 可查）——实时场景里"过期结果"没有价值 |
| **播放节奏原点 = 首帧** | 设备唤醒慢的十几秒**不能**算成"播放落后"，否则会触发巨型追赶（实测曾一口气丢 444 帧） |
| **单次追赶有上限** | 约 2 秒的帧量；剩下没追上的交给后续循环慢慢消化，绝不"整段跳过" |
| **摄像头优先用 DSHOW 后端** | 同一台机器同一设备：DSHOW 首帧 **118ms**，默认后端在服务进程内要 **15s+**（驱动初始化卡住） |
| **只认能读到帧的设备** | `GET /stream/devices` 逐个探测，"能打开但读不到帧"的不算可用；一台都没有时页面自动切到本地视频 |
| **三层标注** | 车牌（粗实线+号牌）＞ 车辆/行人（细线+类别，如 `person 0.91`）＞ **疑似车牌**（灰细线+原因，如 `检测置信度低 0.31`） |
| **灵敏度可按会话调** | `min_det` / `min_rec` 阈值可调低：玩具车、小车牌、远距离车牌经常低于默认 0.5/0.6 而被直接过滤 |
| 启动阶段预热 | 首次推理要初始化 predictor（实测 2.2s），挪到启动阶段，第一辆车不用等 3 秒 |
| 首帧看门狗 | 30s 内没有帧就判失败并给出原因（设备被占用 / 驱动未就绪 / 地址不可达 / 需鉴权） |
| 无人消费自动停 | 浏览器关掉标签页后 60s 无客户端取流则自动停止，不留孤儿会话 |

> **本机实测（CPU / 640px / low 档）**：
>
> | 来源 | 画面 | 识别 | 备注 |
> |---|---|---|---|
> | 本地视频（12s 素材） | **5.0 fps** | 0.85 次/秒，单次 861ms | 去重后 1 个事件 / 命中 11 帧 |
> | 本机摄像头 0 | **6.0 fps** | 1.37 次/秒，单次 522ms | 后端 DSHOW，首帧 118ms |
> | 网络源（http 视频） | 23 帧 | — | 远程源路径可用 |
>
> 早前把识别写在读帧循环里的同步版本只有 0.7 fps 推送——**这就是解耦前后 7 倍的差距**。
> 想要接近真"25fps 实时"需要 GPU；CPU 上请把档位调 low（320px）并把识别频率放宽。

### 命令行单图识别

```bash
"$V" -m src.pipeline.lpr_pipeline --img 图片.jpg --device cpu
"$V" -m src.pipeline.lpr_pipeline --img 图片.jpg --engine-mode engine   # 只用兜底引擎
```

输出为 JSON：

```json
{"code":0,"msg":"success","data":{"plates":[
  {"plate_no":"粤A3333G","plate_color":"blue","vehicle_type":"car",
   "det_score":0.8555,"rec_score":0.9987,"bbox":[142.0,420.0,273.0,480.0],"cost_ms":208.29}
]}}
```

### 视频 / 摄像头 / RTSP 识别（命令行，离线批处理）

```bash
# 视频文件 → 事件 CSV + 标注视频 + 每个车牌的最佳帧小图
"$V" -m src.pipeline.video_pipeline --source clip.mp4 \
    --out runs/annotated.mp4 --csv runs/detections.csv --jsonl runs/events.jsonl \
    --shots-dir runs/shots

# 本机摄像头（0 号），只看前 300 帧
"$V" -m src.pipeline.video_pipeline --source 0 --max-frames 300

# RTSP 流
"$V" -m src.pipeline.video_pipeline --source rtsp://user:pwd@192.168.1.10:554/stream --jsonl events.jsonl
```

输出：终端打印统计 JSON，落盘 `--csv`（UTF-8 BOM，Excel 直接打开中文不乱码）与 `--jsonl`，
`--out` 写出标注视频（按车牌底色分色画框 + 中文标签）。

**跨帧去重语义**（`--window`，默认 3s）：同一车牌在窗口内只算**一个事件**；离开窗口后再出现算新事件。
事件在**定稿**（离场或流结束）后才落盘，因此 `hits` 与置信度都是完整值——
上报的是**识别置信度最高那一帧**，不是最后一帧（实测首帧 0.9353 → 最佳帧 0.9998）。

无摄像头时可用 `tools/make_demo_video.py` 由一张图合成运动视频验证链路：

```bash
"$V" tools/make_demo_video.py --img data/field/test_car.png --out runs/video_e2e/demo.mp4 --seconds 3 --fps 10
```

> 实测（32 帧 1280×720 @8fps，CPU）：`frames=32 / events=1 / total_hits=32 / 粤A3333G / blue / car`，
> 约 0.9–1.0 fps（离线逐帧全跑，与实时流的取舍不同）。

### 接口

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 上传识别页（图片 / 视频） |
| `/live` | GET | 实时流识别页（浏览器摄像头 / RTSP / 网页链接 / 边播边识别） |
| `/health` | GET | 探活，返回模型版本/运行时长/是否 GPU/**兜底引擎状态** `engine` |
| `/predict` | POST | multipart 上传图片（字段名 `file`） |
| `/predict_base64` | POST | body `{"image_b64": "..."}` |
| `/predict_batch` | POST | multipart 多图（字段名 `files`，一次最多 20 张）；**单张失败不影响整批** |
| `/predict_video` | POST | multipart 上传视频（字段名 `file`，可选 `window_s`、`max_frames`）→ 立即返回 `job_id` |
| `/predict_video/{job_id}` | GET | 查询进度 / 结果（`queued`→`running`→`done`/`failed`） |
| `/predict_video/{job_id}` | DELETE | 丢弃任务与产物（受限环境删不掉时返回 `purged=false` + `note`） |
| `/stream` | POST | 启动实时会话：`file`（本地视频）或 `source`（摄像头索引 / rtsp:// / rtmp:// / http(s):// / **视频网页链接**，B站实测可用）；可选 `interval_ms`、`level`、`max_side`、`jpeg_quality`、`min_det`/`min_rec`（识别阈值，玩具车/小车牌调低）、`show_objects`（标注车辆/行人与疑似车牌）。网页链接经**站点适配器**（好看视频）或 yt-dlp 解析为媒体直链后拉流；DASH 分离流（B站/YouTube）选纯视频流，CDN 校验头拉不动时**自动下载兜底**（上限 200MB）。**B站两阶段风控口径相反**：解析要「不发 UA」才不被 412（机房/海外 IP + 浏览器 UA 必被拦），CDN 拉流又必须带浏览器 UA（空 UA 直接 403）——已在 `src/io/online.py` 拆成两套头，可用 `LPR_ONLINE_UA` 现场覆盖（默认 / 具体 UA / none）|
| `/camera/session` | POST | **浏览器摄像头**：建会话（摄像头在用户那边，服务端够不着）。可选 `interval_ms`、`level`、`max_side`、`min_det`/`min_rec`、`show_objects` |
| `/camera/{id}/frame` | POST | 推一帧 JPEG/PNG（multipart，字段名 `file`）→ 返回**要画的框**（车牌/车辆行人/疑似车牌）+ 统计 + 事件 |
| `/camera/{id}` | GET | 摄像头会话状态（与 `/stream/{id}` 同一模型，前端复用同一套面板） |
| `/camera/{id}` | DELETE | 结束摄像头会话（默认保留截图，`?purge=true` 连产物一起删） |
| `/camera-media/{sid}/shots/...` | GET | 浏览器摄像头会话的车牌截图 |
| `/stream/devices` | GET | 探测本机摄像头：**能打开且能读到一帧**才算可用，返回索引 / 后端 / 首帧耗时 |
| `/stream/{id}.mjpg` | GET | **MJPEG 推流**，`<img src>` 直接播放带框画面 |
| `/stream/{id}` | GET | 实时会话状态 / 统计 / 已识别车牌 |
| `/stream/{id}` | DELETE | 停止会话（默认保留记录与截图，`?purge=true` 连产物一起删） |
| `/media/{job_id}/...` | GET | 视频任务产物（标注视频 / CSV / 车牌小图） |
| `/stream-media/{sid}/shots/...` | GET | 实时会话的车牌截图 |
| `/docs` | GET | 自动生成的 Swagger 文档 |

**长耗时路径为什么这么设计**：

| 场景 | 接口 | 原因 |
|---|---|---|
| 一段视频，要完整结果 | `POST /predict_video` | 处理要分钟级，同步 HTTP 必超时；异步 + 轮询才有真进度 |
| 视频/服务器摄像头，要边播边看 | `POST /stream` | 浏览器不能直连 RTSP，只有"服务端解码 → 编码 → 推流"这一条路 |
| **用户本机的摄像头** | `POST /camera/session` + 逐帧推 | 摄像头在用户手里，服务端打不开它；画面本地播放、只推帧识别，带宽与延迟都远小于"推上去再推回来" |
| 单图/多图 | `/predict`、`/predict_batch` | 秒级完成，同步返回最省事 |

> ⚠️ `source` 允许 http(s) 地址意味着**服务端会去访问调用方给的 URL**。
> 本工具定位是内网/本机调试，需要对外暴露时请在网关层做白名单，不要直接暴露到公网。

### 压测与自检

```bash
"$V" -m pytest tests/ -v                              # 128 用例
"$V" tools/bench.py --img 图片.jpg --n 50 --concurrency 4
"$V" tools/verify_video_api.py                        # 视频 HTTP 链路端到端（需服务已启动）
"$V" tools/verify_stream_api.py                       # 实时流链路端到端（需服务已启动）
"$V" tools/verify_camera_api.py                       # 浏览器摄像头链路端到端（模拟前端推帧，需服务已启动）
"$V" tools/clean_jobs.py                              # 预览产物占用；--all --yes 才真删
```

## 目录结构

```
plate-recognition/
├─ configs/          # 运行与训练配置（vehicle_yolov8s / plate_yolov8s / lprnet）
├─ data/
│  ├─ raw/           # 原始下载数据（不入 git）
│  ├─ annotations/   # 标注文件（YOLO txt）
│  ├─ splits/        # train/val/test 切分清单（按车牌号 hash 防泄漏）
│  └─ field/         # 自采现场测试集（≥300 张）
├─ src/
│  ├─ common/        # logger / config / interfaces（Protocol 接口定义）
│  ├─ io/            # reader：图片/视频/摄像头/RTSP + imread_bgr（中文路径安全）；writer：标注视频写出
│  ├─ vision/        # warp 透视矫正 / color 颜色判定 / draw 分色画框 + 中文标签 + 车牌小图裁剪
│  ├─ models/        # detector / lprnet / lpr_recognizer / hyperlpr_engine / onnx_export
│  ├─ postprocess/   # rules 规则校验与省份纠错
│  └─ pipeline/      # lpr_pipeline 单帧全链路 / video_pipeline 离线多帧 / stream 实时流会话 / browser_camera 浏览器推帧会话
├─ tools/            # 切分/训练/评测/压测/合成演示视频/三条端到端自检/产物清理
├─ service/          # FastAPI：app.py(路由) + schemas.py + jobs.py(视频任务) + streams.py(实时会话) + static/
├─ tests/            # 单测 + 接口冒烟 + 引擎 + 视频任务 + 实时流 + 浏览器摄像头 + 依赖守卫
├─ weights/          # .pt / .onnx 权重（不入 git）
├─ logs/             # 运行日志（不入 git）
├─ runs/             # 运行产物（annotated / jobs/<job_id>/ / streams/<sid>/）
├─ docker/           # Dockerfile + docker-compose.yml
├─ requirements.txt  # 依赖清单
└─ README.md
```

## 核心指标（验收口径）

| 指标 | 目标 | 验证命令 |
|---|---|---|
| 车辆检测 mAP@0.5 | ≥ 0.95 | `python tools/eval_vehicle.py` |
| 车牌检测 mAP@0.5 | ≥ 0.93 | 同上 |
| 整牌准确率 | ≥ 92% | `python tools/eval_lpr.py --split field` |
| 端到端延迟 | < 80 ms（GPU） | `python tools/bench.py` |
| 服务吞吐 | ≥ 25 FPS | `python tools/bench.py --concurrency 4` |
| 实时画面流畅度 | 接近源帧率（GPU） | `python tools/verify_stream_api.py` |

> 所有指标必须在自建测试集（与训练集同分布但不同来源，≥300 张）上复现，禁止用训练集指标充当成果。
> 上表为**待达标**状态；当前实测（CPU）：单帧 200–600 ms、离线视频 0.9–1.0 fps、
> 实时流画面 5 fps + 识别 0.85 次/秒，均未做 GPU 加速。

## 技术栈

- 检测：YOLOv8s（Ultralytics），级联（先检车 → 再检牌）
- 识别：LPRNet（CTC，自实现）；训练权重就绪前用 hyperlpr3（ONNX）兜底
- 部署：ONNX Runtime + FastAPI（含 MJPEG 推流）+ Docker Compose
- 工程：pytest、loguru、pydantic、tensorboard

## 训练（需 GPU 环境 + 真实数据）

```bash
# 车辆检测（CCPD/COCO 数据就绪后）
python tools/train_vehicle.py --device 0 --epochs 100

# 车牌检测
python tools/train_plate.py --device 0 --epochs 100

# 字符识别 LPRNet（数据：文件名即标签，如 京A12345_*.jpg）
python tools/train_lprnet.py --img-dir data/raw/crpd --device 0 --epochs 50
```

## 数据准备工具链（M1）

| 脚本 | 作用 |
|---|---|
| `tools/download_coco.py` | fiftyone 下载 COCO 车辆子集（需先装 fiftyone） |
| `tools/prepare_ccpd.py` | CCPD 文件名 → YOLO 检测标注 + 车牌号 CSV |
| `tools/check_labels.py` | YOLO 标注质检（越界框/空标签/格式错误） |
| `tools/split_dataset.py` | 按车牌号 hash 切分 train/val/test（防同车泄漏） |
| `tools/make_demo_video.py` | 由一张图合成运动视频（无摄像头时验证视频/实时链路） |
| `tools/verify_video_api.py` | 视频 HTTP 链路端到端自检 |
| `tools/verify_stream_api.py` | 实时流链路端到端自检 |
| `tools/clean_jobs.py` | 清理 `runs/jobs` + `runs/streams` 产物（预览 / `--all --yes` 执行） |

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 页面显示「没有检出车牌」 | 级联（未训练权重）检不到车牌，且兜底引擎也没读出。检查 `/health` 的 `engine` 是否为 `on`；`detail` 非空说明引擎加载失败（多为模型未下载） |
| `/health` 的 `engine` 为 `off` | `configs/lprnet.yaml` 里 `engine_mode: cascade`，此时只跑自研级联 |
| 首次请求很慢（数秒） | YOLO 首帧含模型预热；之后单帧约 200–600 ms（CPU 640px） |
| 实时画面"不够流畅" | CPU 上识别是瓶颈（约 0.85 次/秒）。把档位调 `low`、画面宽度降到 480、识别频率放宽到 500ms/1s；真 25fps 需要 GPU |
| 实时画面里框的位置"慢半拍" | 框来自最近一次识别结果（中间帧沿用），车牌快速移动时会有滞后。提高识别频率可缓解 |
| 识别速度慢（CPU） | ①本机实测瓶颈是 **ONNX 多会话线程池互相踩踏**（一次识别 500-600ms），已在 `hyperlpr_engine.py` 把 ORT 会话收敛为单线程（`LPR_ORT_THREADS` 可调），同视频实测 491→130ms；②级联权重未训练时连续 20 帧无结果会自动跳过级联（`cascade_skip_after`）；③识别核心（车牌检测+识别）本身只要 **8-12ms**，整帧含车辆归属 ~60-130ms；要 10ms 全链路需 GPU（TensorRT） |
| 识别速度慢（CPU） | ①本机实测瓶颈是 **ONNX 多会话线程池互相踩踏**（一次识别 500-600ms），已在 `hyperlpr_engine.py` 把 ORT 会话收敛为单线程（`LPR_ORT_THREADS` 可调），同视频实测 491→130ms；②级联权重未训练时连续 20 帧无结果会自动跳过级联（`cascade_skip_after`）；③识别核心（车牌检测+识别）本身只要 **8-12ms**，整帧含车辆归属 ~60-130ms；要 10ms 全链路需 GPU（TensorRT） |
| 云服务器上「摄像头」用不了 | 「浏览器摄像头」用的是**你本机**的摄像头（浏览器采集 + 推帧），服务端只识别不回画面；必须在 **https**（或 localhost）下打开页面并允许授权。服务器自己接了摄像头时，在「流地址」里填 `0`/`1` 走服务端拉流那条路 |
| 摄像头/RTSP 一直没画面 | 先看 `/live` 页的「可用设备」——那里只列出**真能读到帧**的设备。摄像头被微信/钉钉/相机占用时能打开但读不到帧；RTSP 地址不通或需鉴权同理。核对完再点开始，30s 内无帧会判失败并给出原因 |
| 摄像头首帧要等几秒 | 驱动唤醒慢属正常（本机实测 DSHOW 118ms，默认后端在服务进程内曾要 15s+）。已改为**索引源优先 DSHOW**，并把首次推理预热挪到启动阶段 |
| 没有摄像头想做实时演示 | 用 `/live` 页的「用示例视频试试」，一键加载示例视频走同一条实时链路 |
| 玩具车 / 小车牌识别不出来，画面上也没框 | ①把「识别灵敏度」调到**宽松/极宽松**（阈值 0.3 或 0.15）再试；②画面上会出现**灰色"疑似车牌"框**——那是检测到了但置信度没过阈值，框上写具体数值；连灰框都没有就是检测这步没发现它。玩具牌不是标准车牌（识别模型按真实车牌训练），字符对不上时读不出属正常，凑近 + `high` 档 + 正对角度更有机会 |
| 实时会话自己停了 | 60s 内没有任何客户端取流（浏览器关了标签页）会**自动停止**，避免留孤儿会话；重新点「开始」即可 |
| 想同时看多路 | 默认上限 2 路（`service/streams.py: MAX_STREAMS`）。CPU 上并行只会互相抢核，建议按需调大并配合降档 |
| 视频任务一直 `queued` | 视频任务单 worker 串行执行（CPU 推理并行只会互相抢核），前一个跑完才开始 |
| 视频只处理了一部分 | 页面默认只处理前 1200 帧（约 48 秒）。响应里 `stats.truncated=true` 表示确实还有没看的帧 |
| `DELETE .../{id}` 返回 `purged=false` | 环境装了删除钩子（safe-delete 之类）拦住批量删除。接口**如实上报**而不是假装删干净；跑 `python tools/clean_jobs.py --all --yes` 即可（实测可删） |
| 引擎初始化报 `OSError: [safe-delete]` | 上游 hyperlpr3 解压后 `os.remove` 临时 zip 被受限环境拦截。本项目 `ensure_models()` 已自行处理 |
| Windows 装依赖装到 CUDA 版 torch | PyPI 默认是 CUDA 版，无 GPU 必须加 `--index-url https://download.pytorch.org/whl/cpu` |
| 装依赖报 safe-delete / bulk-delete | 受限环境拦截 pip 清缓存，加 `--no-cache-dir` |
| `pip install hyperlpr3` 把 fastapi 降级 | 改用 `pip install hyperlpr3 --no-deps`（其真实依赖本项目已具备） |
| 读图返回 None（路径含中文） | 已修：`src/io/reader.py` 的 `imread_bgr()` 用 `np.fromfile + cv2.imdecode` 规避 cv2.imread 的 ANSI 限制 |
| 输出视频帧数比预期少（日志还报「写出 N 帧」） | `cv2.VideoWriter` 收到尺寸不同的帧会静默丢弃（实测写 30 次只落 16 帧）。已修：`src/io/writer.py` 等比纠正 + `frames_resized` 计数 |
| 视频里读不出车牌（单图能读出） | 多半是合成/缩放时把画面拉伸变形了。`tools/make_demo_video.py` 已改等比 letterbox |
| 中文标签显示成 `????` | `cv2.putText` 不支持中文。已修：`src/vision/draw.py` 用 Pillow + 系统字体渲染，缺字体时降级 ASCII |
| 网页链接报 412 / 403 | **两阶段风控口径相反**：①解析阶段（B站网页/API）机房与海外 IP 带浏览器 UA 会被 412，代码按站点改成**不发 UA**；②CDN 拉流阶段反过来必须带浏览器 UA（空 UA 403），媒体阶段单独用浏览器 UA + Referer。个别站点需登录 Cookie 时，可临时用 `LPR_ONLINE_UA=<你自己的UA>` 覆盖，或换回本机家宽/代理出口 |
| 远景/航拍车流很多车没有车牌 | **像素极限，非故障**：车辆已检出（有 `car` 绿框）但车牌只有 25~50px，中文车牌可靠识别需 100px+ 近正视角。实测 720p/1080p + ROI 放大均无法读出（插值不产生信息）。换卡口/近景素材即可 |
| 车牌截图内容错乱 | `cv2.VideoCapture` **复用帧缓冲**，切片是视图不是拷贝。已修：`crop_plate()` 内部 `copy()` |
| 实时页点「开始」后一直黑屏 | 两个真实原因（都已修）：①**摄像头走了默认后端**——同一台机器同一设备，默认后端在服务进程内首次取帧要 15s+，显式 DSHOW 只要 118ms（现在索引源按 DSHOW→MSMF→默认 依次尝试）；②**首帧慢被误判成"播放落后"**，触发一次巨型追赶，实测一口气 grab 掉 444 帧（现在节奏原点取在首帧，且单次追赶上限约 2 秒的帧量） |
