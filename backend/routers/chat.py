"""
聊天问答路由
实现基于 RAG 的问答功能，支持 Server-Sent Events (SSE) 流式返回
"""
import json
import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from backend.dependencies import get_app_config
from core.rag_chain import RAGChain
from utils.logger import get_logger

logger = get_logger()
router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    """聊天请求模型"""
    question: str = Field(..., description="用户的提问内容")
    kb_id: str = Field(default="default", description="知识库 ID")
    history: Optional[list] = Field(default_factory=list, description="对话历史消息列表")
    api_key: Optional[str] = Field(default=None, description="大模型 API Key")
    base_url: Optional[str] = Field(default=None, description="大模型 Base URL")
    model_name: Optional[str] = Field(default=None, description="使用的模型名称")


@router.post("")
async def chat_endpoint(
    request: Request,
    chat_req: ChatRequest,
    config: dict = Depends(get_app_config)
):
    """
    RAG 问答接口，使用 SSE 流式返回回答
    """
    logger.info(f"收到聊天请求: {chat_req.question[:20]}... [KB: {chat_req.kb_id}, History: {len(chat_req.history)}]")
    
    try:
        # 初始化 RAGChain
        rag = RAGChain(
            config=config,
            api_key=chat_req.api_key,
            base_url=chat_req.base_url,
            model_name=chat_req.model_name
        )
        
        async def event_generator():
            # 获取同步的生成器，传入 history
            generator = rag.query_stream(
                chat_req.question, 
                kb_id=chat_req.kb_id,
                history=chat_req.history
            )
            
            # TODO: 目前 langchain 的 stream 是同步的，如果要完全异步，可以考虑使用 astream。
            # 为了兼容同步生成器在 async 协程中运行不阻塞主线程，这里我们使用简单的阻塞迭代。
            # 在实际生产高并发场景下，应使用 asyncio.to_thread 包装或更换为 astream。
            for item in generator:
                # 检查客户端是否已断开连接
                if await request.is_disconnected():
                    logger.info("客户端已断开连接")
                    break
                
                # 发送 SSE 数据块
                yield {
                    "event": item["type"],
                    "data": json.dumps(item["data"], ensure_ascii=False)
                }
                
                # 让出控制权，以便处理其他请求
                await asyncio.sleep(0.01)
                
            # 发送结束标志
            yield {
                "event": "finish",
                "data": json.dumps({"text": ""}, ensure_ascii=False)
            }
            logger.info("流式传输完成")

        return EventSourceResponse(event_generator())

    except Exception as e:
        logger.error(f"问答过程出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
