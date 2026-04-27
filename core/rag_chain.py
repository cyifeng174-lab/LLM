"""
DocMind 在线检索问答链
封装 "检索 → Prompt 组装 → LLM 生成" 的完整流程
使用 LangChain LCEL 表达式语法构建链
"""
from typing import Generator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from core.data_pipeline import DataPipeline
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger()


class RAGChain:
    """
    在线检索问答链
    
    负责：
    1. 从向量数据库检索相关文档片段
    2. 组装 Prompt（系统提示 + 上下文 + 用户问题）
    3. 调用 LLM 生成回答
    4. 返回回答及来源溯源信息
    """
    
    def __init__(self, config: dict = None, api_key: str = None, 
                 base_url: str = None, model_name: str = None):
        """
        初始化问答链
        
        Args:
            config: 配置字典，默认从 config.yaml 加载
            api_key: LLM API Key（优先级高于配置文件）
            base_url: LLM API Base URL（优先级高于配置文件）
            model_name: LLM 模型名称（优先级高于配置文件）
        """
        if config is None:
            config = get_config()
        
        self.config = config
        llm_config = config.get("llm", {})
        retriever_config = config.get("retriever", {})
        
        # LLM 参数（传入参数优先于配置文件）
        self.api_key = api_key
        self.base_url = base_url or llm_config.get("base_url", "")
        self.model_name = model_name or llm_config.get("model_name", "qwen-plus")
        self.temperature = llm_config.get("temperature", 0.1)
        self.max_tokens = llm_config.get("max_tokens", 2048)
        
        # 检索参数
        self.top_k = retriever_config.get("top_k", 3)
        self.search_type = retriever_config.get("search_type", "similarity")
        
        # Prompt 模板
        system_prompt = config.get("prompt", {}).get("system", "")
        if not system_prompt:
            system_prompt = (
                "你是一个专业的智能助手。"
                "请参考以下检索到的上下文，回答用户的问题。"
                "如果你不知道答案，就明确表示无法在文档中找到，不要试图编造。\n\n"
                "上下文：\n{context}"
            )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 数据管道（用于获取向量数据库）
        self.pipeline = DataPipeline(config)
        
        logger.info(
            f"RAGChain 初始化完成 | "
            f"model={self.model_name}, top_k={self.top_k}"
        )
    
    def _get_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        if not self.api_key:
            raise ValueError("未提供 API Key，请在侧边栏输入或设置环境变量 DOCMIND_API_KEY")
        
        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    @staticmethod
    def _format_docs(docs) -> str:
        """将检索到的文档列表格式化为纯文本"""
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "")
            page_info = f" (第{page}页)" if page else ""
            formatted.append(
                f"【片段{i+1}】来源: {source}{page_info}\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)
    
    def query(self, question: str, kb_id: str = "default", history: list = None) -> dict:
        """
        执行 RAG 问答 (支持多轮)
        """
        history = history or []
        logger.info(f"收到问题: {question[:50]}... | 知识库: {kb_id} | 历史轮数: {len(history)}")
        
        # 1. 如果有历史记录，重构问题
        processed_query = question
        llm = self._get_llm()
        
        if history:
            condense_prompt = ChatPromptTemplate.from_template(
                "给定以下对话历史和用户的一个新问题，请将其重构为一个独立的问题，使其在没有对话历史的情况下也能被理解。只需输出重构后的问题文本。\n\n对话历史：\n{chat_history}\n新问题：{input}"
            )
            # 格式化历史
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]]) # 只取最近5轮
            condense_chain = condense_prompt | llm | StrOutputParser()
            try:
                processed_query = condense_chain.invoke({"chat_history": history_str, "input": question})
                logger.info(f"问题重构结果: {processed_query}")
            except Exception as e:
                logger.warning(f"重构问题失败，将使用原问题进行检索。错误: {e}")
                processed_query = question

        # 2. 获取检索器
        vectorstore = self.pipeline.get_vectorstore(kb_id)
        from core.retrievers import HybridRetriever
        retriever = HybridRetriever.from_kb(
            kb_id=kb_id,
            vectorstore=vectorstore,
            bm25_dir=self.pipeline.bm25_dir,
            top_k=self.top_k,
            recall_k=10
        )
        
        # 3. 构建 LCEL 链
        rag_chain = (
            {
                "context": retriever | self._format_docs,
                "input": RunnablePassthrough(),
            }
            | self.prompt
            | llm
            | StrOutputParser()
        )
        
        # 执行检索
        source_docs = retriever.invoke(processed_query)
        
        # 执行问答链
        answer = rag_chain.invoke(processed_query)
        
        # 构造来源信息
        sources = []
        for doc in source_docs:
            sources.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", None),
            })
        
        return {"answer": answer, "sources": sources}
    
    def query_stream(self, question: str, kb_id: str = "default", history: list = None) -> Generator:
        """
        流式问答 (支持多轮)
        """
        history = history or []
        logger.info(f"流式问答: {question[:50]}... | 知识库: {kb_id} | 历史轮数: {len(history)}")
        
        processed_query = question
        llm = self._get_llm()
        
        if history:
            condense_prompt = ChatPromptTemplate.from_template(
                "给定以下对话历史和用户的一个新问题，请将其重构为一个独立的问题，使其在没有对话历史的情况下也能被理解。只需输出重构后的问题文本。\n\n对话历史：\n{chat_history}\n新问题：{input}"
            )
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
            condense_chain = condense_prompt | llm | StrOutputParser()
            try:
                processed_query = condense_chain.invoke({"chat_history": history_str, "input": question})
                logger.info(f"流式问题重构: {processed_query}")
            except Exception as e:
                logger.warning(f"重构问题失败，将使用原问题进行检索。错误: {e}")
                processed_query = question

        # 获取检索器
        vectorstore = self.pipeline.get_vectorstore(kb_id)
        from core.retrievers import HybridRetriever
        retriever = HybridRetriever.from_kb(
            kb_id=kb_id,
            vectorstore=vectorstore,
            bm25_dir=self.pipeline.bm25_dir,
            top_k=self.top_k,
            recall_k=10
        )
        
        # 构建 LCEL 链
        rag_chain = (
            {
                "context": retriever | self._format_docs,
                "input": RunnablePassthrough(),
            }
            | self.prompt
            | llm
            | StrOutputParser()
        )
        
        # 执行检索以获取来源信息
        source_docs = retriever.invoke(processed_query)
        sources = []
        for doc in source_docs:
            sources.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", None),
            })
            
        # 首次产出来源信息
        yield {"type": "sources", "data": sources}
        
        # 流式输出文本块
        for chunk in rag_chain.stream(question):
            yield {"type": "chunk", "data": chunk}
