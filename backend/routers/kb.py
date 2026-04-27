"""
知识库管理路由
"""
from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_data_pipeline
from core.data_pipeline import DataPipeline
from utils.logger import get_logger

logger = get_logger()
router = APIRouter(prefix="/kb", tags=["Knowledge Base"])


@router.get("/list")
async def list_knowledge_bases(
    pipeline: DataPipeline = Depends(get_data_pipeline)
):
    """
    获取所有的知识库列表
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=pipeline.chroma_dir)
        collections = client.list_collections()
        # collections 是一个列表，每个元素有 name 属性
        kb_names = [col.name for col in collections]
        
        return {
            "status": "success",
            "data": {
                "total": len(kb_names),
                "knowledge_bases": kb_names
            }
        }
    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    pipeline: DataPipeline = Depends(get_data_pipeline)
):
    """
    删除指定的知识库
    """
    success = pipeline.delete_kb(kb_id)
    if success:
        return {"status": "success", "message": f"知识库 {kb_id} 已删除"}
    else:
        raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 删除失败或不存在")
