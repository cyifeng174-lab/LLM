"""
DocMind 离线数据处理管道
封装完整的 "文件解析 → 文本切分 → 向量化 → 存储" 流程
"""
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from core.embeddings import get_embedding_model
from utils.file_parser import parse_file
from utils.config_loader import get_config, get_abs_path
from utils.logger import get_logger

logger = get_logger()


class DataPipeline:
    """
    离线数据处理管道
    
    负责：
    1. 解析上传的文件（TXT/PDF/Word/Markdown）
    2. 将文档切分为合适大小的 chunks
    3. 向量化并存入 Chroma 向量数据库
    """
    
    def __init__(self, config: dict = None):
        """
        初始化数据管道
        
        Args:
            config: 配置字典，默认从 config.yaml 加载
        """
        if config is None:
            config = get_config()
        
        self.config = config
        
        # 初始化文本切分器
        splitter_config = config.get("splitter", {})
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=splitter_config.get("chunk_size", 500),
            chunk_overlap=splitter_config.get("chunk_overlap", 50),
            separators=splitter_config.get("separators", ["\n\n", "\n", "。", "！", "？", "；", " ", ""]),
        )
        
        # Chroma 持久化目录
        self.chroma_dir = get_abs_path(
            config.get("paths", {}).get("chroma_db", "data/chroma_db")
        )
        os.makedirs(self.chroma_dir, exist_ok=True)
        
        # 上传文件保存目录
        self.upload_dir = get_abs_path(
            config.get("paths", {}).get("uploads", "data/uploads")
        )
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # BM25 索引目录
        self.bm25_dir = get_abs_path(
            config.get("paths", {}).get("bm25_index", "data/bm25_index")
        )
        os.makedirs(self.bm25_dir, exist_ok=True)
        
        logger.info(
            f"DataPipeline 初始化完成 | "
            f"chunk_size={splitter_config.get('chunk_size', 500)}, "
            f"chunk_overlap={splitter_config.get('chunk_overlap', 50)}"
        )
    
    def process_file(
        self,
        file_path: str,
        kb_id: str = "default",
    ) -> dict:
        """
        处理单个文件：解析 → 切分 → 向量化 → 存入 Chroma
        
        Args:
            file_path: 文件路径
            kb_id: 知识库 ID，用于 Chroma collection 隔离
        
        Returns:
            处理结果字典: {
                "file_name": str,
                "chunks_count": int,
                "collection": str,
            }
        """
        file_name = os.path.basename(file_path)
        logger.info(f"开始处理文件: {file_name} → 知识库: {kb_id}")
        
        # Step 1: 解析文件
        logger.info("Step 1/3: 解析文件...")
        docs = parse_file(file_path)
        logger.info(f"  解析得到 {len(docs)} 个原始文档段")
        
        # Step 2: 切分文档
        logger.info("Step 2/3: 切分文档...")
        chunks = self.text_splitter.split_documents(docs)
        
        # 为每个 chunk 注入额外 metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["kb_id"] = kb_id
            chunk.metadata["chunk_index"] = i
        
        logger.info(f"  切分得到 {len(chunks)} 个文本块")
        
        # Step 3: 向量化并存入 Chroma
        logger.info("Step 3/3: 向量化并存入数据库...")
        embedding_model = get_embedding_model(self.config)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=kb_id,
            persist_directory=self.chroma_dir,
        )
        
        # Step 4: 构建 BM25 索引 (Week 2 优化)
        logger.info("Step 4/4: 构建 BM25 混合检索索引...")
        try:
            import jieba
            import pickle
            from rank_bm25 import BM25Okapi
            
            # 获取当前知识库中的所有文本和元数据
            all_data = vectorstore.get(include=["documents", "metadatas"])
            all_docs = all_data.get("documents", [])
            all_metadatas = all_data.get("metadatas", [])
            
            if all_docs:
                # 分词
                tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]
                bm25 = BM25Okapi(tokenized_corpus)
                
                # 保存模型及对应的元数据，以便检索时溯源
                bm25_data = {
                    "model": bm25,
                    "documents": all_docs,
                    "metadatas": all_metadatas
                }
                
                bm25_path = os.path.join(self.bm25_dir, f"{kb_id}.pkl")
                with open(bm25_path, "wb") as f:
                    pickle.dump(bm25_data, f)
                logger.info(f"  BM25 索引已保存至 {bm25_path}，包含 {len(all_docs)} 个文档块")
        except Exception as e:
            logger.error(f"  构建 BM25 索引失败: {e}")
            
        result = {
            "file_name": file_name,
            "chunks_count": len(chunks),
            "collection": kb_id,
        }
        
        logger.info(
            f"文件处理完成: {file_name} | "
            f"共 {len(chunks)} 个块 | "
            f"知识库: {kb_id}"
        )
        
        return result
    
    def get_vectorstore(self, kb_id: str = "default") -> Chroma:
        """
        获取指定知识库的 Chroma 向量数据库实例
        
        Args:
            kb_id: 知识库 ID
        
        Returns:
            Chroma 向量数据库实例
        """
        embedding_model = get_embedding_model(self.config)
        
        vectorstore = Chroma(
            collection_name=kb_id,
            embedding_function=embedding_model,
            persist_directory=self.chroma_dir,
        )
        
        return vectorstore
    
    def delete_kb(self, kb_id: str) -> bool:
        """
        删除指定知识库
        
        Args:
            kb_id: 知识库 ID
        
        Returns:
            是否删除成功
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_dir)
            
            # 删除 Chroma Collection
            try:
                client.delete_collection(kb_id)
            except Exception as e:
                logger.warning(f"Chroma 中不存在集合 {kb_id} 或删除失败: {e}")
                
            # 删除 BM25 索引
            bm25_path = os.path.join(self.bm25_dir, f"{kb_id}.pkl")
            if os.path.exists(bm25_path):
                os.remove(bm25_path)
                
            logger.info(f"知识库及相关索引已彻底删除: {kb_id}")
            return True
        except Exception as e:
            logger.error(f"删除知识库失败: {kb_id}, 错误: {e}")
            return False
    
    def save_upload(self, file_name: str, file_content: bytes) -> str:
        """
        保存上传的文件到 uploads 目录
        
        Args:
            file_name: 文件名
            file_content: 文件二进制内容
        
        Returns:
            保存后的文件绝对路径
        """
        save_path = os.path.join(self.upload_dir, file_name)
        with open(save_path, "wb") as f:
            f.write(file_content)
        logger.info(f"文件已保存: {save_path}")
        return save_path
