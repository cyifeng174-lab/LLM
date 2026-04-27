"""
文件上传路由
处理文档上传、解析并存入知识库
"""
import os
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException

from backend.dependencies import get_data_pipeline
from core.data_pipeline import DataPipeline
from utils.logger import get_logger
from utils.file_parser import SUPPORTED_EXTENSIONS

logger = get_logger()
router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form("default"),
    pipeline: DataPipeline = Depends(get_data_pipeline)
):
    """
    上传文档并处理存入知识库
    """
    logger.info(f"收到文件上传请求: {file.filename}, 知识库: {kb_id}")
    
    # 验证文件格式
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件格式: {ext}。支持的格式: {supported}"
        )
        
    try:
        # 读取文件内容
        content = await file.read()
        
        # 保存到本地 uploads 目录
        file_path = pipeline.save_upload(file.filename, content)
        
        # 执行数据管道处理
        result = pipeline.process_file(file_path, kb_id=kb_id)
        
        return {
            "status": "success",
            "message": "文档处理完成",
            "data": result
        }
    except Exception as e:
        logger.error(f"处理上传文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
