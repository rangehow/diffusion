# tasks/mmlu.py
from typing import Dict, List, Optional
import datasets
from .base import BaseTask, register_task, TaskConfig # 注意这里的相对导入
from ..sampler import BaseContextSampler, get_sampler

@register_task("mmlu")
class MMLUTask(BaseTask):
    TASK_NAME = "mmlu"
    QUESTION_COL = "question"
    OPTIONS_COL = "choices"
    ANSWER_COL = "answer"
    
    # MMLU 所有主题列表
    MMLU_SUBJECTS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    def __init__(self, config: TaskConfig):
        # MMLUTask 有特殊的 sampler 逻辑，所以我们在这里初始化它
        self.samplers: Dict[str, BaseContextSampler] = {}
        super().__init__(config)

    def _load_dataset(self) -> datasets.Dataset:
        return datasets.load_dataset('cais/mmlu', 'all', split='test', cache_dir=self.config.local_dir)

    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        if self.config.num_fewshot > 0:
            return datasets.load_dataset('cais/mmlu', 'all', split='dev', cache_dir=self.config.local_dir)
        return None

    def _setup_sampler(self) -> Optional[BaseContextSampler]:
        """
        为 MMLU 的每个 subject 单独创建并设置一个 Sampler。
        这个方法会覆盖 BaseTask 的通用 sampler 设置。
        """
        if self.config.num_fewshot > 0 and self.fewshot_dataset:
            print(f"[{self.TASK_NAME}] Setting up subject-specific samplers...")
            for subject in self.MMLU_SUBJECTS:
                # 筛选出当前 subject 的 few-shot 数据
                subject_fewshot_data = self.fewshot_dataset.filter(lambda x: x["subject"] == subject)
                
                if len(subject_fewshot_data) < self.config.num_fewshot:
                     print(f"Warning: Not enough few-shot examples for MMLU subject '{subject}'. "
                           f"Required: {self.config.num_fewshot}, Available: {len(subject_fewshot_data)}")

                sampler_kwargs = {"fewshot_dataset": subject_fewshot_data, "seed": self.config.sampler_seed}
                self.samplers[subject] = get_sampler(self.config.sampler_name, **sampler_kwargs)
            
            print(f"[{self.TASK_NAME}] Successfully created {len(self.samplers)} subject-specific samplers.")
        # 返回 None，因为我们不使用单一的 self.sampler
        return None
    
    # 新增：一个静态/类方法，专门用于格式化单个 MMLU 样本
    # 这个方法完美复刻了 qwen_script 中的 format_example 函数
    @staticmethod
    def _format_example(example: Dict, include_answer: bool = True) -> str:
        """
        将单个 MMLU 样本（字典格式）格式化为字符串。
        
        Args:
            example: 一个样本字典，应包含 'question', 'choices', 'answer' 键。
            include_answer: 是否在末尾包含 "Answer: [letter]"。
        """
        choices_letters = ["A", "B", "C", "D"]
        
        prompt = "Question: " + example["question"]
        for i, choice_text in enumerate(example["choices"]):
            prompt += f'\n{choices_letters[i]}. {choice_text}'

        if include_answer:
            answer_index = int(example["answer"])
            correct_letter = choices_letters[answer_index]
            prompt += "\nAnswer: " + correct_letter
        else:
            prompt += "\nAnswer:"
        return prompt

    # 新增：辅助函数，与 qwen_script 中的 format_subject 一致
    @staticmethod
    def _format_subject(subject: str) -> str:
        # qwen_script 中没有 capitalize，但通常 MMLU prompt 会大写首字母
        return " ".join(word for word in subject.split("_"))

    def process(self) -> datasets.Dataset:
        """
        重写 process 方法，将所有格式化逻辑放在 Task 内部。
        """
        dataset_to_finalize = self._load_dataset()
        if not self.config.text_only and self.config.tokenizer is None:
            raise ValueError("Tokenizer must be provided when text_only is False.")

        def _format_instance(batch: Dict[str, List]) -> Dict[str, List]:
            if self.config.text_only:
                new_batch = {"context_text": [], "continuation_text": [], "is_correct": [], "group_id": [], "task_name": []}
            else:
                new_batch = {"input_ids": [], "continuation_ids": [], "is_correct": [], "group_id": [], "task_name": [], "continuation_len": [], "continuation_char_len": []}

            choices_letters = ["A", "B", "C", "D"]

            for i in range(len(batch[self.QUESTION_COL])):
                subject = batch['subject'][i]
                
                # --- MMLU 特定的 Prompt 构建逻辑 ---

                # 1. 构建 Prompt 头部
                prompt_header = f"The following are multiple choice questions (with answers) about {self._format_subject(subject)}.\n\n"

                # 2. 从 Sampler 获取原始 Few-shot 样本
                fewshot_samples = []
                if self.samplers:
                    sampler_for_subject = self.samplers.get(subject)
                    if sampler_for_subject:
                        # 调用 sampler 的新方法 get_samples
                        fewshot_samples = sampler_for_subject.get_samples(self.config.num_fewshot)

                # 3. 使用内部方法格式化 Few-shot 样本
                formatted_fewshot_list = [
                    self._format_example(sample, include_answer=True) for sample in fewshot_samples
                ]
                fewshot_prompt_str = "\n\n".join(formatted_fewshot_list)
                if formatted_fewshot_list:
                    fewshot_prompt_str += "\n\n"

                # 4. 格式化当前 Test 样本
                current_test_sample = {
                    "question": batch[self.QUESTION_COL][i],
                    "choices": batch[self.OPTIONS_COL][i],
                    # answer 是为了满足 _format_example 的接口，但不会被使用
                    "answer": batch[self.ANSWER_COL][i] 
                }
                test_question_block = self._format_example(current_test_sample, include_answer=False)

                # 5. 组合成最终的上下文 (Context)
                prompt_str = prompt_header + fewshot_prompt_str + test_question_block
                
                # --- 后续逻辑与之前类似 ---
                group_id = batch["id"][i]
                original_answer_index = int(batch[self.ANSWER_COL][i])
                options = batch[self.OPTIONS_COL][i]

                for j, opt_text in enumerate(options):
                    is_correct = 1 if j == original_answer_index else 0
                    
                    # continuation 应该是选项字母，以对齐 qwen_script 的单 token logit 评估方法
                    continuation_str = " " + choices_letters[j]

                    if self.config.text_only:
                        new_batch["context_text"].append(prompt_str)
                        new_batch["continuation_text"].append(continuation_str)
                    else:
                        tokenizer = self.config.tokenizer
                        input_ids = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
                        if tokenizer.bos_token_id is not None:
                            input_ids = [tokenizer.bos_token_id] + input_ids
                        
                        continuation_ids = tokenizer(continuation_str, add_special_tokens=False)['input_ids']
                        
                        new_batch["input_ids"].append(input_ids)
                        new_batch["continuation_ids"].append(continuation_ids)
                        new_batch["continuation_len"].append(len(continuation_ids))
                        new_batch["continuation_char_len"].append(len(continuation_str.lstrip()))
                    
                    new_batch["is_correct"].append(is_correct)
                    new_batch["group_id"].append(group_id)
                    new_batch["task_name"].append(f"{self.TASK_NAME}:{subject}")
            
            return new_batch

        if "id" not in dataset_to_finalize.column_names:
            dataset_to_finalize = dataset_to_finalize.add_column("id", range(len(dataset_to_finalize)))

        # 按 subject 排序可以提高 filter 的效率
        dataset_to_finalize = dataset_to_finalize.sort("subject")
        return dataset_to_finalize.map(_format_instance, num_proc=16, batched=True, remove_columns=dataset_to_finalize.column_names, desc=f"[{self.TASK_NAME}] Processing with subject-specific few-shot")
