# tasks/winogrande.py

import datasets
from .base import BaseTask, register_task
from typing import Dict, List

@register_task("winogrande")
class WinograndeTask(BaseTask):
    """
    Winogrande 是一个大规模的常识推理数据集，专注于解决代词指代问题。
    任务形式是句子补全，模型需要选择两个选项中哪一个能更好地填补句子中的空白。
    
    lm_eval 配置参考:
    - dataset_path: winogrande
    - dataset_name: winogrande_xl
    - validation_split: validation
    - training_split: train (for few-shot)
    - output_type: multiple_choice
    """
    TASK_NAME = "winogrande"
    
    # Winogrande 是一个典型的句子补全任务。
    # "question" 是句子的前半部分，"options" 是可能的结尾。
    IS_SENTENCE_COMPLETION = True

    def _load_dataset(self) -> datasets.Dataset:
        """
        加载用于评估的验证集 (validation split)。
        """
        return self._load_and_process("validation")

    def _load_fewshot_dataset(self) -> datasets.Dataset:
        """
        加载用于 few-shot 示例的训练集 (train split)。
        """
        return self._load_and_process("train")

    def _preprocess(self, doc: Dict) -> Dict:
        """
        对单个数据点进行预处理，将其转换为框架所需的格式。

        原始数据格式:
        {
            "sentence": "The city councilmen refused the demonstrators a permit because _ feared violence.",
            "option1": "they",
            "option2": "The city councilmen",
            "answer": "1"
        }

        转换后格式:
        {
            "question": "The city councilmen refused the demonstrators a permit because",
            "options": ["they feared violence.", "The city councilmen feared violence."],
            "answer": 0
        }
        """
        
        sentence = doc["sentence"]
        try:
            split_point = sentence.index("_")
            # `question` 是句子中 '_' 之前的部分
            question_text = sentence[:split_point].strip()
            # `rest_of_sentence` 是句子中 '_' 之后的部分，这部分是两个选项共有的后缀
            rest_of_sentence = sentence[split_point + 1:].strip()
        except ValueError:
            # 如果句子中没有 "_"，虽然不符合数据集规范，但做一个兼容处理
            question_text = sentence.strip()
            rest_of_sentence = ""
            print(f"Warning: '_' not found in sentence: {sentence}")

        option1 = doc["option1"]
        option2 = doc["option2"]

        # 构造完整的句子结尾作为选项。
        # 如果 `rest_of_sentence` 为空，则选项就是 `option1/2` 本身。
        # 如果 `rest_of_sentence` 不为空，则需要拼接。
        continuation1 = f"{option1} {rest_of_sentence}".strip()
        continuation2 = f"{option2} {rest_of_sentence}".strip()
        
        options = [continuation1, continuation2]

        # 将答案 "1" 或 "2" 转换为 0-indexed 的整数
        answer_index = int(doc["answer"]) - 1

        return {
            self.QUESTION_COL: question_text,
            self.OPTIONS_COL: options,
            self.ANSWER_COL: answer_index
        }

    def _load_and_process(self, split: str) -> datasets.Dataset:
        """
        一个辅助函数，用于加载指定 split 的数据集并应用预处理。
        """
        dataset = datasets.load_dataset(
            path="winogrande",
            name="winogrande_xl",
            split=split
        )
        
        dataset = dataset.map(
            self._preprocess,
            num_proc=16,
            remove_columns=dataset.column_names,
            desc=f"[{self.TASK_NAME}] Preprocessing {split} split",
            load_from_cache_file=False 
        )
        return dataset