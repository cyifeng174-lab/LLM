"""
混合检索与 Reranker 精排模块
提供 Chroma(向量) + BM25(词频) 双路召回，以及 RRF 融合与二次精排。
"""
import os
import pickle
import jieba
from typing import List, Dict, Any
from pydantic import Field

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from sentence_transformers import CrossEncoder

from utils.logger import get_logger

logger = get_logger()


# 全局单例 Reranker 模型
_GLOBAL_RERANKER = None

class HybridRetriever(BaseRetriever):
    """
    自研混合检索器 (Hybrid Retriever)
    1. 并行调用 Chroma 稠密检索和 BM25 稀疏检索
    2. 使用 RRF (Reciprocal Rank Fusion) 合并召回结果
    3. 调用 BAAI Reranker 进行二次精确打分与截断
    """
    
    vectorstore: Any = Field(description="Chroma 向量数据库实例")
    bm25_model: Any = Field(default=None, description="BM25Okapi 实例")
    bm25_docs: List[str] = Field(default_factory=list, description="BM25对应的纯文本块")
    bm25_metadatas: List[Dict] = Field(default_factory=list, description="BM25对应的元数据")
    
    reranker: Any = Field(default=None, description="CrossEncoder 模型")
    
    top_k: int = Field(default=3, description="最终返回的文档数量")
    recall_k: int = Field(default=10, description="单路召回的数量 (用于 RRF 合并前)")
    
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_kb(cls, kb_id: str, vectorstore: Any, bm25_dir: str, top_k: int = 3, recall_k: int = 10):
        """
        工厂方法，初始化检索器并加载所需模型
        """
        global _GLOBAL_RERANKER
        
        # 1. 加载 BM25 模型
        bm25_model = None
        bm25_docs = []
        bm25_metadatas = []
        
        bm25_path = os.path.join(bm25_dir, f"{kb_id}.pkl")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    bm25_data = pickle.load(f)
                    bm25_model = bm25_data.get("model")
                    bm25_docs = bm25_data.get("documents", [])
                    bm25_metadatas = bm25_data.get("metadatas", [])
                logger.info(f"成功加载 BM25 索引: {kb_id}，包含 {len(bm25_docs)} 个文档块")
            except Exception as e:
                logger.error(f"加载 BM25 索引失败: {e}")
        else:
            logger.warning(f"未找到 BM25 索引文件: {bm25_path}，将降级为纯向量检索")
            
        # 2. 加载 Reranker 模型 (全局单例懒加载)
        if _GLOBAL_RERANKER is None:
            try:
                logger.info("首次加载 Reranker 模型 BAAI/bge-reranker-v2-m3，这可能需要一点时间...")
                _GLOBAL_RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
                logger.info("Reranker 模型全局单例加载成功")
            except Exception as e:
                logger.error(f"加载 Reranker 模型失败，将跳过精排阶段: {e}")
            
        return cls(
            vectorstore=vectorstore,
            bm25_model=bm25_model,
            bm25_docs=bm25_docs,
            bm25_metadatas=bm25_metadatas,
            reranker=_GLOBAL_RERANKER,
            top_k=top_k,
            recall_k=recall_k
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        核心检索逻辑
        """
        # 1. Chroma 向量召回
        logger.info(f"HybridRetriever 启动 -> Query: {query}")
        vector_docs = self.vectorstore.similarity_search(query, k=self.recall_k)
        
        # 2. BM25 关键词召回
        bm25_docs_result = []
        if self.bm25_model:
            tokenized_query = list(jieba.cut(query))
            # get_top_n 返回的是源文本列表
            top_texts = self.bm25_model.get_top_n(tokenized_query, self.bm25_docs, n=self.recall_k)
            
            # 将源文本重新映射组装为 Document 对象
            for text in top_texts:
                try:
                    idx = self.bm25_docs.index(text)
                    meta = self.bm25_metadatas[idx]
                    bm25_docs_result.append(Document(page_content=text, metadata=meta))
                except ValueError:
                    continue
        
        logger.info(f"双路召回完成: 向量={len(vector_docs)}条, BM25={len(bm25_docs_result)}条")
        
        # 3. RRF (Reciprocal Rank Fusion) 融合
        # RRF_Score = 1 / (k + rank)
        rrf_k = 60
        fused_scores = {}
        doc_map = {}
        
        # 融合向量结果
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.page_content  # 简单起见，以内容作为唯一键（实际最好用 UUID）
            doc_map[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
            
        # 融合 BM25 结果
        for rank, doc in enumerate(bm25_docs_result):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
            
        # 按 RRF 分数排序
        fused_docs = [doc_map[doc_id] for doc_id, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
        
        # 限制精排候选数量
        candidates = fused_docs[:self.recall_k]
        logger.info(f"RRF 融合后候选文档数: {len(candidates)}")
        
        # 4. Reranker 二次精排
        if self.reranker and candidates:
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = self.reranker.predict(pairs)
            
            # 将得分绑定到文档
            scored_candidates = list(zip(candidates, scores))
            # 按得分降序排列
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 截取最终的 Top K
            final_docs = [doc for doc, score in scored_candidates[:self.top_k]]
            logger.info(f"Reranker 精排完成，最高分: {scored_candidates[0][1]:.4f}，返回 Top-{len(final_docs)}")
        else:
            final_docs = candidates[:self.top_k]
            
        return final_docs
