# tasks/bbh.py

from typing import Dict, List, Optional
import datasets
from .base import BaseTask, register_task, TaskConfig
from ..sampler import BaseContextSampler, get_sampler

@register_task("bbh")
class BBHTask(BaseTask):
    TASK_NAME = "bbh"
    # --- 主要修改点 1: 更新列名以匹配实际数据集 ---
    QUESTION_COL = "input"
    ANSWER_COL = "target"  # 正确答案在 'target' 列中
    # OPTIONS_COL 已移除，因为数据集中没有提供选项列表
    IS_SENTENCE_COMPLETION = False # BBH 是问答/推理任务，不是简单的句子补全
    
    # BBH 所有子任务列表
    BBH_SUBJECTS = [
        'boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa',
        'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton',
        'logical_deduction_five_objects', 'logical_deduction_seven_objects', 
        'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two',
        'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
        'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding',
        'temporal_sequences', 'tracking_shuffled_objects_five_objects', 
        'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
        'web_of_lies', 'word_sorting'
    ]

    def __init__(self, config: TaskConfig):
        # BBHTask 有特殊的 sampler 逻辑，所以我们在这里初始化它
        self.samplers: Dict[str, BaseContextSampler] = {}
        super().__init__(config)

    def _load_dataset(self) -> datasets.Dataset:
        """加载BBH数据集，合并所有子任务"""
        try:
            # 分别加载每个子任务，然后合并
            all_datasets = []
            for subject in self.BBH_SUBJECTS:
                # BBH的测试集就是它的验证集
                dataset = datasets.load_dataset("lukaemon/bbh", subject, split="test")
                # 添加任务名称列，以便后续识别
                dataset = dataset.add_column("task", [subject] * len(dataset))
                all_datasets.append(dataset)
            
            if not all_datasets:
                raise ValueError("Could not load any BBH sub-tasks")
                
            # 合并所有子任务
            combined_dataset = datasets.concatenate_datasets(all_datasets)
            print(f"Combined all BBH sub-tasks: {len(combined_dataset)} examples total")
            return combined_dataset
        except Exception as e:
            print(f"Error loading BBH dataset: {e}")
            if self.config.local_dir:
                try:
                    dataset = datasets.load_from_disk(self.config.local_dir)
                    return dataset
                except Exception as e2:
                    raise ValueError(f"Could not load BBH dataset from local directory {self.config.local_dir}: {e2}")
            raise ValueError(f"Could not load BBH dataset: {e}")

    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        """加载BBH的few-shot示例数据集"""
        if self.config.num_fewshot > 0:
            # 对于BBH，通常使用 'test' split 作为 few-shot 来源，因为它没有单独的 'train' split
            # 这是一个常见的评估实践，从评估集中抽样作为上下文示例
            return self._load_dataset()
        return None

    def _setup_sampler(self) -> Optional[BaseContextSampler]:
        """
        为 BBH 的每个子任务单独创建并设置一个 Sampler。
        """
        if self.config.num_fewshot > 0 and self.fewshot_dataset:
            print(f"[{self.TASK_NAME}] Setting up task-specific samplers...")
            for subject in self.BBH_SUBJECTS:
                subject_fewshot_data = self.fewshot_dataset.filter(lambda x: x["task"] == subject)
                
                if len(subject_fewshot_data) < self.config.num_fewshot:
                     print(f"Warning: Not enough few-shot examples for BBH task '{subject}'. "
                           f"Required: {self.config.num_fewshot}, Available: {len(subject_fewshot_data)}")

                sampler_kwargs = {"fewshot_dataset": subject_fewshot_data, "seed": self.config.sampler_seed}
                self.samplers[subject] = get_sampler(self.config.sampler_name, **sampler_kwargs)
            
            print(f"[{self.TASK_NAME}] Successfully created {len(self.samplers)} task-specific samplers.")
        return None
    
    # --- 主要修改点 2: 重写 _format_example 以处理 'input' 和 'target' 字符串 ---
    @staticmethod
    def _format_example(example: Dict, include_answer: bool = True) -> str:
        """
        将单个 BBH 样本格式化为字符串。
        
        Args:
            example: 一个样本字典，应包含 'input', 'target' 键。
            include_answer: 是否包含答案。
        """
        question = example["input"]
        prompt = f"Question: {question}\nAnswer:"
        
        if include_answer:
            answer = example["target"]
            prompt += f" {answer}" # 添加一个空格以更好地分隔
        
        return prompt

    def process(self) -> datasets.Dataset:
        """
        重写 process 方法，将所有格式化逻辑放在 Task 内部。
        """
        dataset_to_finalize = self._load_dataset()
        if not self.config.text_only and self.config.tokenizer is None:
            raise ValueError("Tokenizer must be provided when text_only is False.")

        # --- 主要修改点 3: 重写核心处理逻辑 _format_instance ---
        def _format_instance(batch: Dict[str, List]) -> Dict[str, List]:
            if self.config.text_only:
                # 对于生成式任务，我们通常只有 context 和 continuation
                new_batch = {"context_text": [], "continuation_text": [], "is_correct": [], "group_id": [], "task_name": []}
            else:
                new_batch = {"input_ids": [], "continuation_ids": [], "is_correct": [], "group_id": [], "task_name": [], "continuation_len": [], "continuation_char_len": []}

            for i in range(len(batch["input"])):
                task = batch['task'][i]
                
                # --- BBH 特定的 Prompt 构建逻辑 ---

                # 1. 从 Sampler 获取原始 Few-shot 样本
                fewshot_samples = []
                if self.samplers:
                    sampler_for_task = self.samplers.get(task)
                    if sampler_for_task:
                        fewshot_samples = sampler_for_task.get_samples(self.config.num_fewshot)

                # 2. 使用新的 _format_example 格式化 Few-shot 样本
                formatted_fewshot_list = [
                    self._format_example(sample, include_answer=True) for sample in fewshot_samples
                ]
                fewshot_prompt_str = "\n\n".join(formatted_fewshot_list)
                if formatted_fewshot_list:
                    fewshot_prompt_str += "\n\n"

                # 3. 格式化当前 Test 样本（不包含答案）
                current_test_sample = {
                    "input": batch["input"][i]
                }
                test_question_block = self._format_example(current_test_sample, include_answer=False)

                # 4. 组合成最终的上下文 (Context)
                prompt_str = fewshot_prompt_str + test_question_block
                
                # 5. continuation 就是正确的答案
                continuation_str = f" {batch[self.ANSWER_COL][i]}" # 在答案前加一个空格，这是常见做法
                
                group_id = batch["id"][i]
                # 由于我们只提供正确答案作为continuation，所以is_correct总是1
                is_correct = 1

                if self.config.text_only:
                    new_batch["context_text"].append(prompt_str)
                    new_batch["continuation_text"].append(continuation_str)
                else:
                    tokenizer = self.config.tokenizer
                    input_ids = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
                    if tokenizer.bos_token_id is not None:
                        input_ids = [tokenizer.bos_token_id] + input_ids
                    
                    continuation_ids = tokenizer(continuation_str, add_special_tokens=False)['input_ids']
                    
                    if self.config.model_type == 'discrete_diffusion' and tokenizer.eos_token_id is not None:
                        continuation_ids = continuation_ids + [tokenizer.eos_token_id]
                    
                    new_batch["input_ids"].append(input_ids)
                    new_batch["continuation_ids"].append(continuation_ids)
                    new_batch["continuation_len"].append(len(continuation_ids))
                    new_batch["continuation_char_len"].append(len(continuation_str.lstrip()))
                
                new_batch["is_correct"].append(is_correct)
                new_batch["group_id"].append(group_id)
                new_batch["task_name"].append(f"{self.TASK_NAME}:{task}")
            
            return new_batch

        if "id" not in dataset_to_finalize.column_names:
            # 生成唯一的ID，如果数据集中没有提供
            ids = [f"bbh-{i}" for i in range(len(dataset_to_finalize))]
            dataset_to_finalize = dataset_to_finalize.add_column("id", ids)

        # 按 task 排序可以提高 filter 的效率
        dataset_to_finalize = dataset_to_finalize.sort("task")
        
        # 移除不再需要的 'target' 列，以避免 map 函数的列冲突
        dataset_to_finalize = dataset_to_finalize.remove_columns([self.ANSWER_COL])

        return dataset_to_finalize.map(
            _format_instance, 
            num_proc=16, 
            batched=True, 
            remove_columns=dataset_to_finalize.column_names, 
            desc=f"[{self.TASK_NAME}] Processing with task-specific few-shot"
        )