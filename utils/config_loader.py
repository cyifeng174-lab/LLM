"""
DocMind 配置加载器
读取 config.yaml 并提供全局单例访问
"""
import os
import yaml

# 项目根目录 = docmind/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

_config_cache = None


def load_config(config_path: str = None) -> dict:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径，默认为项目根目录下的 config.yaml
    
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = CONFIG_PATH
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def get_config() -> dict:
    """
    获取全局配置单例
    首次调用时加载配置文件，后续调用返回缓存
    
    Returns:
        配置字典
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def get_abs_path(relative_path: str) -> str:
    """
    将配置中的相对路径转为绝对路径
    
    Args:
        relative_path: 相对于项目根目录的路径
    
    Returns:
        绝对路径
    """
    return os.path.join(PROJECT_ROOT, relative_path)
