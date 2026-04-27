"""
DocMind — 企业文档智能问答平台
Streamlit 前端入口 (前后端分离版)

Day 5-7 重构：
本脚本现已成为纯客户端代码，不再直接导入任何模型或数据处理逻辑。
所有的业务操作（上传、获取知识库列表、RAG 问答）全部通过 requests 库向 FastAPI 后端发送 HTTP 请求。
"""
import os
import sys
import json
import requests
import streamlit as st

# 将项目根目录加入 sys.path，仅为了使用公共日志和配置工具
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import get_config
from utils.logger import setup_logger, get_logger
from utils.file_parser import SUPPORTED_EXTENSIONS

# ============================================================
# 初始化与常量
# ============================================================
setup_logger()
logger = get_logger()
config = get_config()

# FastAPI 后端地址
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocMind - 智能知识库问答",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# API 客户端封装
# ============================================================
def get_knowledge_bases():
    """向后端拉取知识库列表"""
    try:
        resp = requests.get(f"{BACKEND_URL}/kb/list", timeout=3)
        if resp.status_code == 200:
            return resp.json()["data"]["knowledge_bases"]
    except Exception as e:
        logger.error(f"无法连接到后端服务获取知识库列表: {e}")
    return []


# ============================================================
# 侧边栏：配置 + 文件上传
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.caption("企业文档智能问答平台 (分离架构版)")
    st.divider()
    
    # 状态提示
    try:
        requests.get(f"{BACKEND_URL}/", timeout=1)
        st.success("🟢 后端服务已连接")
    except:
        st.error("🔴 无法连接后端，请确保 Uvicorn 已启动")
    
    # API 配置
    st.markdown("### ⚙️ 模型配置")
    llm_config = config.get("llm", {})
    api_key = st.text_input(
        "API Key",
        type="password",
        help="请输入大模型 API Key",
    )
    base_url = st.text_input(
        "Base URL",
        value=llm_config.get("base_url", ""),
        help="兼容 OpenAI 格式的 API 地址",
    )
    model_name = st.text_input(
        "模型名称",
        value=llm_config.get("model_name", "qwen-plus"),
    )
    
    st.divider()
    
    # 知识库管理
    st.markdown("### 📚 知识库管理")
    
    kb_list = get_knowledge_bases()
    if not kb_list:
        kb_list = ["default"]
    
    kb_id = st.selectbox(
        "选择现有知识库",
        options=kb_list,
        index=0,
    )
    
    # 新建知识库
    new_kb_name = st.text_input("或创建新知识库", placeholder="输入新知识库名称")
    if new_kb_name.strip():
        kb_id = new_kb_name.strip()
        
    # 删除当前选中的知识库
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ 删除知识库", use_container_width=True):
            if kb_id == "default":
                st.error("不能删除 default 知识库")
            else:
                try:
                    resp = requests.delete(f"{BACKEND_URL}/kb/{kb_id}")
                    if resp.status_code == 200:
                        st.success(f"已删除 {kb_id}")
                        st.rerun()
                    else:
                        st.error("删除失败")
                except Exception as e:
                    st.error(f"连接失败: {e}")
    
    with col2:
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    # 文件上传
    supported_types = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_file = st.file_uploader(
        "📁 上传新文档至该知识库",
        type=supported_types,
        help=f"支持格式: {', '.join(SUPPORTED_EXTENSIONS)}",
    )
    
    process_btn = st.button("🚀 处理文档并建库", type="primary", use_container_width=True)

# ============================================================
# 主区域标题
# ============================================================
st.title("🧠 DocMind 智能问答")
st.markdown("上传文档到左侧知识库，然后在下方对话框提问。前端将通过 API 请求调用后端进行问答。")

# ============================================================
# 文档处理逻辑
# ============================================================
if process_btn and uploaded_file:
    with st.spinner("📄 正在提交文件到后端处理..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"kb_id": kb_id}
            
            response = requests.post(f"{BACKEND_URL}/upload", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json().get("data", {})
                st.session_state.current_kb = kb_id
                st.success(
                    f"✅ 文档处理完成！\n\n"
                    f"- 文件: **{result.get('file_name')}**\n"
                    f"- 切分为 **{result.get('chunks_count')}** 个块\n"
                    f"- 知识库: **{result.get('collection')}**"
                )
            else:
                st.error(f"❌ 后端处理失败: {response.text}")
        except Exception as e:
            st.error(f"❌ 请求失败: {str(e)}")

elif process_btn and not uploaded_file:
    st.warning("⚠️ 请先上传一个文档文件")

# ============================================================
# 聊天问答界面
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 查看来源知识片段"):
                for i, src in enumerate(message["sources"]):
                    page_info = f" | 第{src.get('page')}页" if src.get("page") else ""
                    st.markdown(f"**片段 {i+1}** — 来源: `{src.get('source')}`{page_info}")
                    st.info(src.get("content"))

# 接收新问题
if question := st.chat_input("基于知识库提问..."):
    if not api_key:
        st.warning("⚠️ 请先在侧边栏输入 API Key")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            current_kb = st.session_state.get("current_kb", kb_id)
            
            payload = {
                "question": question,
                "kb_id": current_kb,
                "history": st.session_state.messages[-5:], # 发送最近 5 条历史
                "api_key": api_key,
                "base_url": base_url,
                "model_name": model_name
            }
            
            try:
                # 使用 stream=True 发起 SSE 请求
                response = requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True)
                
                if response.status_code != 200:
                    st.error(f"❌ 请求失败: {response.text}")
                else:
                    sources_data = []
                    
                    def stream_parser():
                        global sources_data
                        current_event = None
                        
                        # 解析 SSE 格式: "event: ...\ndata: ...\n\n"
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            line_str = line.decode('utf-8')
                            
                            if line_str.startswith('event: '):
                                current_event = line_str[7:]
                            elif line_str.startswith('data: '):
                                data_str = line_str[6:]
                                try:
                                    data_obj = json.loads(data_str)
                                    
                                    if current_event == "sources":
                                        sources_data = data_obj
                                    elif current_event == "chunk":
                                        yield data_obj
                                except json.JSONDecodeError:
                                    pass
                    
                    # 1. 使用 st.write_stream 渲染打字机效果
                    full_answer = st.write_stream(stream_parser())
                    
                    # 2. 如果后端返回了来源数据，则显示折叠面板
                    if sources_data:
                        with st.expander("📚 查看来源知识片段"):
                            for i, src in enumerate(sources_data):
                                page_info = f" | 第{src.get('page')}页" if src.get("page") else ""
                                st.markdown(f"**片段 {i+1}** — 来源: `{src.get('source')}`{page_info}")
                                st.info(src.get("content"))
                                
                    # 3. 保存到会话历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "sources": sources_data,
                    })

            except Exception as e:
                st.error(f"❌ 无法连接到后端服务: {str(e)}")
