# MV-01 车牌识别服务镜像（CPU 版）
FROM python:3.11-slim

WORKDIR /app

# 先拷依赖清单再装依赖（利用层缓存：代码改动不重装 torch）
# 注意顺序：requirements 必须先 COPY 再 install——装依赖时它才在镜像里
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps hyperlpr3
# hyperlpr3 必须 --no-deps：其声明依赖 opencv-python（覆盖 headless，缺 libxcb 崩溃）
# 和 fastapi==0.92（把本项目 fastapi 降级）。真实依赖已由上面 requirements 覆盖

COPY . .

# 权重与日志挂载目录
RUN mkdir -p /app/weights /app/logs

EXPOSE 8000

# 计划书 §10：多 worker 部署用 gunicorn+uvicorn（Linux 下可用）
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
