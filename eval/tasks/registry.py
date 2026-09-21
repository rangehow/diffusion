# eval/tasks/registry.py
"""
Task registry for automatic task discovery and registration.
"""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseTask


class TaskRegistry:
    """
    Registry for evaluation tasks.
    
    Tasks are registered using the @register_task decorator and can be
    retrieved by name using the get() method.
    """
    
    _tasks: Dict[str, Type["BaseTask"]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a task class.
        
        Args:
            name: Unique name for the task
            
        Returns:
            Decorator function
            
        Raises:
            ValueError: If task name is already registered
            
        Example:
            @TaskRegistry.register("my_task")
            class MyTask(BaseTask):
                ...
        """
        def decorator(task_class: Type["BaseTask"]):
            if name in cls._tasks:
                raise ValueError(
                    f"Task '{name}' is already registered by {cls._tasks[name].__name__}"
                )
            cls._tasks[name] = task_class
            return task_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type["BaseTask"]:
        """
        Get a task class by name.
        
        Args:
            name: Name of the task
            
        Returns:
            Task class
            
        Raises:
            ValueError: If task is not registered
        """
        if name not in cls._tasks:
            available = ", ".join(sorted(cls._tasks.keys()))
            raise ValueError(
                f"Unknown task: '{name}'. Available tasks: {available}"
            )
        return cls._tasks[name]
    
    @classmethod
    def list_tasks(cls) -> list:
        """Return list of all registered task names."""
        return sorted(cls._tasks.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a task is registered."""
        return name in cls._tasks


# Convenience decorator alias
register_task = TaskRegistry.register


def get_task(name: str, config):
    """
    Factory function to instantiate a task.
    
    Args:
        name: Task name
        config: TaskConfig instance
        
    Returns:
        Instantiated task
    """
    task_class = TaskRegistry.get(name)
    return task_class(config)