"""
DocMind FastAPI 后端入口文件
提供前后端分离的 REST API 服务
启动命令: uvicorn backend.main:app --reload
"""
import sys
import os

# 将项目根目录加入 sys.path，以便能够正确导入 core 和 utils 等模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import upload, chat, kb
from utils.logger import setup_logger, get_logger

# 初始化日志系统
setup_logger()
logger = get_logger()
logger.info("正在启动 DocMind FastAPI 后端服务...")

# 创建 FastAPI 应用
app = FastAPI(
    title="DocMind API",
    description="企业级 RAG 知识库问答系统后端 API",
    version="2.0.0",
)

# 配置跨域请求 (CORS) 允许前端请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许所有来源，生产环境应配置具体的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有的 HTTP 方法 (GET, POST, OPTIONS 等)
    allow_headers=["*"],  # 允许所有的 HTTP 请求头
)

# 注册各个模块的路由
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(kb.router)


@app.get("/")
async def root():
    """系统健康检查接口"""
    return {
        "status": "online",
        "service": "DocMind RAG Backend",
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    # 提供快速启动方式：python -m backend.main
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
