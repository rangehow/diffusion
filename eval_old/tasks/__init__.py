# tasks/__init__.py
import os
import importlib
from .base import get_task, TASK_REGISTRY

# 自动发现并导入当前目录下的所有 .py 文件 (除了 __init__.py 和 base.py)
# 这样每个文件中的 @register_task 就会被执行
package_dir = os.path.dirname(__file__)
for filename in os.listdir(package_dir):
    if filename.endswith(".py") and filename not in ["__init__.py", "base.py"]:
        module_name = filename[:-3]
        # 使用 importlib 动态导入模块
        importlib.import_module(f".{module_name}", package=__name__)

# 暴露给外部使用的接口
__all__ = ["get_task", "TASK_REGISTRY"]