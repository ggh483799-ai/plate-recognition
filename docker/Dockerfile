# MV-01 车牌识别服务镜像（CPU 版）
FROM python:3.11-slim

WORKDIR /app

# 先装 CPU 版 torch（Linux 上 PyPI 默认 torch 是 CUDA 版，体积大），再装其余依赖
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# 权重与日志挂载目录
RUN mkdir -p /app/weights /app/logs

EXPOSE 8000

# 计划书 §10：多 worker 部署用 gunicorn+uvicorn（Linux 下可用）
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
