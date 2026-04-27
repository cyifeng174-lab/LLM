"""
DocMind 日志系统
使用 loguru 提供统一的日志记录
"""
import sys
import os

from utils.config_loader import get_config, get_abs_path

try:
    from loguru import logger
    _USE_LOGURU = True
except ImportError:
    import logging
    logger = logging.getLogger("docmind")
    _USE_LOGURU = False


def setup_logger():
    """
    初始化日志系统
    - 控制台输出彩色日志
    - 文件输出，支持轮转和保留策略
    """
    config = get_config()
    log_config = config.get("logging", {})
    
    level = log_config.get("level", "INFO")
    rotation = log_config.get("rotation", "10 MB")
    retention = log_config.get("retention", "7 days")
    
    log_dir = get_abs_path(config.get("paths", {}).get("logs", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "docmind.log")
    
    if _USE_LOGURU:
        # 移除默认 handler，重新配置
        logger.remove()
        
        # 控制台输出
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
        )
        
        # 文件输出
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )
        
        logger.info("日志系统初始化完成 (loguru)")
    else:
        # 回退到标准 logging
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stderr),
                logging.FileHandler(log_file, encoding="utf-8"),
            ],
        )
        logger.info("日志系统初始化完成 (标准 logging)")
    
    return logger


def get_logger():
    """获取全局 logger 实例"""
    return logger
