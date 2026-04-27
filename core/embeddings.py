"""
DocMind Embedding 模型封装
提供全局单例的 Embedding 模型实例
"""
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger()

_embedding_instance = None


def get_embedding_model(config: dict = None) -> HuggingFaceEmbeddings:
    """
    获取 Embedding 模型单例
    
    Args:
        config: 配置字典，默认从 config.yaml 加载
    
    Returns:
        HuggingFaceEmbeddings 实例
    """
    global _embedding_instance
    
    if _embedding_instance is not None:
        return _embedding_instance
    
    if config is None:
        config = get_config()
    
    embed_config = config.get("embedding", {})
    model_name = embed_config.get("model_name", "BAAI/bge-small-zh-v1.5")
    device = embed_config.get("device", "cpu")
    encode_kwargs = embed_config.get("encode_kwargs", {"normalize_embeddings": True})
    
    logger.info(f"加载 Embedding 模型: {model_name} (device={device})")
    
    _embedding_instance = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs,
    )
    
    logger.info("Embedding 模型加载完成")
    return _embedding_instance


def reset_embedding_model():
    """重置 Embedding 模型单例（用于测试或切换模型）"""
    global _embedding_instance
    _embedding_instance = None
