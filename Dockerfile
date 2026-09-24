FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 TZ=Asia/Shanghai

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data

# 默认跑 API；ui / scheduler 由 compose 覆盖 command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
