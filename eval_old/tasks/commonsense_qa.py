# tasks/commonsense_qa.py

import datasets
from .base import BaseTask, TaskConfig, register_task

@register_task("commonsense_qa")
class CommonsenseQATask(BaseTask):
    """
    CommonsenseQA (https://www.tau-nlp.sites.tau.ac.il/commonsenseqa)
    is a multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers.
    """
    
    # --- BaseTask 属性配置 ---
    TASK_NAME: str = "commonsense_qa"
    
    # 告诉 BaseTask 数据集中的列名
    QUESTION_COL: str = "question"
    OPTIONS_COL: str = "options"  # 这是我们通过 map 创建的新列名
    ANSWER_COL: str = "answerKey" # 正确答案的列名是 answerKey，值为 'A', 'B' 等
    
    # 这不是一个句子补全任务，而是一个问答任务
    IS_SENTENCE_COMPLETION: bool = False

    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def _load_dataset(self) -> datasets.Dataset:
        """加载用于评估的验证集 (validation split)。"""
        ds = datasets.load_dataset("tau/commonsense_qa", split="validation")
        # 预处理：将嵌套的 choices['text'] 提取到一个名为 'options' 的新顶层列
        return ds.map(lambda x: {"options": x["choices"]["text"]})

    def _load_fewshot_dataset(self) -> datasets.Dataset:
        """加载用于 few-shot 示例的训练集 (train split)。"""
        if self.config.num_fewshot == 0:
            return None
        ds = datasets.load_dataset("tau/commonsense_qa", split="train")
        # 同样进行预处理
        return ds.map(lambda x: {"options": x["choices"]["text"]})