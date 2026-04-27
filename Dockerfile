FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # 如果需要用到 jieba 和 rank_bm25 却没有写进 requirements
    && pip install --no-cache-dir jieba rank-bm25

# 复制项目文件
COPY . .

# 暴露 FastAPI (8000) 和 Streamlit (8501) 端口
EXPOSE 8000
EXPOSE 8501

# 启动脚本将由 docker-compose 控制
CMD ["bash"]
