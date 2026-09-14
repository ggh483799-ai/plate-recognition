# MV-01 车牌识别服务镜像（CPU 版）
FROM python:3.11-slim

WORKDIR /app

# 先拷依赖清单再装依赖（利用层缓存：代码改动不重装 torch）
# 注意顺序：requirements 必须先 COPY 再 install——装依赖时它才在镜像里
COPY requirements.txt ./
# 系统库双保险：即使某个依赖把完整版 opencv-python 带回来（ultralytics/hyperlpr3 都会），
# cv2 也能 import（缺 libxcb/libgl 是容器里最常见的 ImportError）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps hyperlpr3 \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless
# hyperlpr3 必须 --no-deps：其声明依赖 opencv-python（覆盖 headless，缺 libxcb 崩溃）
# 和 fastapi==0.92（把本项目 fastapi 降级）。真实依赖已由上面 requirements 覆盖
# 最后强制重装 headless：ultralytics 的依赖也会带完整版 opencv，压轴保证 cv2 = headless

COPY . .

# 权重与日志挂载目录
RUN mkdir -p /app/weights /app/logs

EXPOSE 8000

# 计划书 §10：多 worker 部署用 gunicorn+uvicorn（Linux 下可用）
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
