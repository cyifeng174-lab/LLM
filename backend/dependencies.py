"""
FastAPI 依赖注入模块
提供获取配置、数据管道和问答链等实例的方法
"""
from fastapi import Depends, Request

import sys
import os
# 确保可以导入项目根目录的模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import get_config
from core.data_pipeline import DataPipeline


def get_app_config() -> dict:
    """获取全局配置"""
    return get_config()


def get_data_pipeline(config: dict = Depends(get_app_config)) -> DataPipeline:
    """获取 DataPipeline 单例实例"""
    # FastAPI 在每次请求时会调用依赖。
    # DataPipeline 内部没有太多状态，初始化开销主要是读取配置。
    # （如果在高并发下，可以使用 lru_cache 缓存该实例）
    return DataPipeline(config)
