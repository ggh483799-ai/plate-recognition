# MV-01 车牌识别服务镜像（CPU 版）
FROM python:3.11-slim

WORKDIR /app

# 先拷依赖清单再装依赖（利用层缓存：代码改动不重装 torch）
# 注意顺序：requirements 必须先 COPY 再 install——装依赖时它才在镜像里
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# 权重与日志挂载目录
RUN mkdir -p /app/weights /app/logs

EXPOSE 8000

# 计划书 §10：多 worker 部署用 gunicorn+uvicorn（Linux 下可用）
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
