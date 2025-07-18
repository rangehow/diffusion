import datasets
import os
import json  # 改为json
from functools import wraps
from typing import Dict, Callable, Any, Optional
from pathlib import Path

# 全局数据集注册表
_DATASET_REGISTRY: Dict[str, Callable] = {}

# 数据集本地路径配置
_LOCAL_PATHS: Dict[str, str] = {}

def load_dataset_config(config_path: str = "diffusion/dataset_config.json"):  # 改为json后缀
    """
    从配置文件加载数据集本地路径映射
    """
    global _LOCAL_PATHS
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)  # 改为json.load
                _LOCAL_PATHS = config.get('local_paths', {})
                print(f"已加载数据集配置: {len(_LOCAL_PATHS)} 个本地路径")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            _LOCAL_PATHS = {}
    else:
        print(f"配置文件 {config_path} 不存在，将使用默认远程加载")
        _LOCAL_PATHS = {}

def register_dataset(remote_loader: Optional[Callable] = None):
    """
    数据集注册装饰器
    支持本地路径优先加载
    """
    def decorator(func: Callable) -> Callable:
        dataset_name = func.__name__
        
        # 检查是否重名
        if dataset_name in _DATASET_REGISTRY:
            raise ValueError(f"数据集 '{dataset_name}' 已经存在，请使用不同的函数名")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取配置的本地路径
            local_path = _LOCAL_PATHS.get(dataset_name)
            
            # 将本地路径作为参数传递给原函数
            return func(local_path, *args, **kwargs)
        
        # 注册数据集
        _DATASET_REGISTRY[dataset_name] = wrapper
        return wrapper
    
    return decorator

def get_dataset(name: str, *args, **kwargs) -> Any:
    """
    根据名称获取数据集
    """
    if name not in _DATASET_REGISTRY:
        available_datasets = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"数据集 '{name}' 未找到。可用数据集: {available_datasets}")
    
    return _DATASET_REGISTRY[name](*args, **kwargs)

def list_datasets() -> list:
    """
    列出所有已注册的数据集名称
    """
    return list(_DATASET_REGISTRY.keys())

def get_local_path(dataset_name: str) -> Optional[str]:
    """
    获取数据集的本地路径
    """
    return _LOCAL_PATHS.get(dataset_name)

# 初始化时加载配置
load_dataset_config()

# 注册数据集
@register_dataset()
def ultra_fineweb(local_path):
    """加载 Ultra-FineWeb 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'ultra_fineweb': {local_path}")
        try:
            return datasets.load_dataset(local_path,'default')['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'ultra_fineweb'")
    return datasets.load_dataset("openbmb/Ultra-FineWeb",'en')['train']

@register_dataset()
def fineweb_10b(local_path):
    """加载 fineweb-edu-dedup-10b 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'fineweb_10b': {local_path}")
        try:
            return datasets.load_dataset(local_path)['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'fineweb_10b'")
    return datasets.load_dataset("EleutherAI/fineweb-edu-dedup-10b")['train']


@register_dataset()
def common_crawl(local_path):
    """加载 Common Crawl 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'common_crawl': {local_path}")
        try:
            return datasets.load_from_disk(local_path)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'common_crawl'")
    return datasets.load_dataset("common_crawl")

@register_dataset()
def wikipedia(local_path):
    """加载 Wikipedia 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'wikipedia': {local_path}")
        try:
            return datasets.load_from_disk(local_path)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'wikipedia'")
    return datasets.load_dataset("wikipedia", "20220301.en")