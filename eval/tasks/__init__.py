# eval/tasks/__init__.py
"""
Evaluation tasks package.

Tasks are automatically discovered and registered when this package is imported.
"""

import os
import importlib

from .registry import TaskRegistry, register_task, get_task
from .base import BaseTask
from .sampler import (
    BaseSampler,
    FirstNSampler,
    RandomSampler,
    BalancedSampler,
    create_sampler,
)


def _auto_discover_tasks():
    """
    Automatically import all task modules in this package.
    
    This triggers the @register_task decorators to execute,
    populating the TaskRegistry.
    """
    package_dir = os.path.dirname(__file__)
    exclude_files = {"__init__.py", "base.py", "registry.py", "sampler.py"}
    
    for filename in os.listdir(package_dir):
        if filename.endswith(".py") and filename not in exclude_files:
            module_name = filename[:-3]
            importlib.import_module(f".{module_name}", package=__name__)


# Run auto-discovery on import
_auto_discover_tasks()


__all__ = [
    # Registry
    "TaskRegistry",
    "register_task",
    "get_task",
    
    # Base classes
    "BaseTask",
    
    # Samplers
    "BaseSampler",
    "FirstNSampler",
    "RandomSampler",
    "BalancedSampler",
    "create_sampler",
]