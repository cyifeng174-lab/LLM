"""Day 2 模块导入验证脚本"""
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("DocMind Day 2 模块验证")
print("=" * 60)

# 1. 测试 config_loader
print("\n[1/6] 测试 config_loader...")
from utils.config_loader import get_config, get_abs_path
config = get_config()
print(f"  ✅ config.yaml 加载成功")
print(f"     embedding: {config['embedding']['model_name']}")
print(f"     llm: {config['llm']['model_name']}")
print(f"     chunk_size: {config['splitter']['chunk_size']}")

# 2. 测试 logger
print("\n[2/6] 测试 logger...")
from utils.logger import setup_logger, get_logger
setup_logger()
logger = get_logger()
logger.info("日志模块测试消息")
print(f"  ✅ logger 初始化成功")

# 3. 测试 file_parser
print("\n[3/6] 测试 file_parser...")
from utils.file_parser import parse_file, SUPPORTED_EXTENSIONS
print(f"  ✅ file_parser 导入成功")
print(f"     支持格式: {SUPPORTED_EXTENSIONS}")

# 用 test.txt 做实际解析测试
test_file = os.path.join(os.path.dirname(__file__), "test.txt")
if os.path.exists(test_file):
    docs = parse_file(test_file)
    print(f"  ✅ test.txt 解析成功, 得到 {len(docs)} 个 Document")
    print(f"     首段前50字: {docs[0].page_content[:50]}...")

# 4. 测试 embeddings
print("\n[4/6] 测试 embeddings...")
from core.embeddings import get_embedding_model
print(f"  ✅ embeddings 模块导入成功")
# 注意：实际加载模型较慢，这里只测导入

# 5. 测试 data_pipeline
print("\n[5/6] 测试 data_pipeline...")
from core.data_pipeline import DataPipeline
print(f"  ✅ DataPipeline 导入成功")

# 6. 测试 rag_chain
print("\n[6/6] 测试 rag_chain...")
from core.rag_chain import RAGChain
print(f"  ✅ RAGChain 导入成功")

print("\n" + "=" * 60)
print("🎉 所有模块导入验证通过！")
print("=" * 60)
