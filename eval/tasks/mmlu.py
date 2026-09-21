# eval/tasks/mmlu.py
"""
MMLU (Massive Multitask Language Understanding) task.
"""

from typing import Dict, List, Optional
import datasets
from .base import BaseTask
from .registry import register_task
from .sampler import BaseSampler, create_sampler

@register_task("mmlu")
class MMLUTask(BaseTask):
    """
    MMLU multi-subject evaluation task.
    """
    
    TASK_NAME = "mmlu"
    QUESTION_COL = "question"
    OPTIONS_COL = "choices"
    ANSWER_COL = "answer"
    IS_SENTENCE_COMPLETION = False
    
    SUBJECTS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
        'clinical_knowledge', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_medicine',
        'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics',
        'formal_logic', 'global_facts', 'high_school_biology',
        'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography',
        'high_school_government_and_politics', 'high_school_macroeconomics',
        'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology',
        'high_school_statistics', 'high_school_us_history',
        'high_school_world_history', 'human_aging', 'human_sexuality',
        'international_law', 'jurisprudence', 'logical_fallacies',
        'machine_learning', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
        'philosophy', 'prehistory', 'professional_accounting',
        'professional_law', 'professional_medicine', 'professional_psychology',
        'public_relations', 'security_studies', 'sociology',
        'us_foreign_policy', 'virology', 'world_religions',
    ]
    
    CHOICE_LETTERS = ["A", "B", "C", "D"]
    
    def __init__(self, config):
        self.subject_samplers: Dict[str, BaseSampler] = {}
        super().__init__(config)
    
    def _load_dataset(self) -> datasets.Dataset:
        """Load MMLU test set using load_hf_dataset for local fallback."""
        return self.load_hf_dataset(
            'cais/mmlu', 'all',
            split='test',
            cache_dir=self.config.local_dir,
        )
    
    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        """Load MMLU dev set for few-shot."""
        if self.config.num_fewshot == 0:
            return None
        return self.load_hf_dataset(
            'cais/mmlu', 'all',
            split='dev',
            cache_dir=self.config.local_dir,
        )
    
    def _setup_sampler(self) -> Optional[BaseSampler]:
        """Set up subject-specific samplers."""
        if self.config.num_fewshot > 0 and self.fewshot_dataset:
            print(f"[{self.TASK_NAME}] Setting up subject-specific samplers...")
            for subject in self.SUBJECTS:
                subject_data = self.fewshot_dataset.filter(
                    lambda x: x["subject"] == subject
                )
                if len(subject_data) < self.config.num_fewshot:
                    print(f"  Warning: {subject} has only {len(subject_data)} examples")
                
                self.subject_samplers[subject] = create_sampler(
                    self.config.sampler_type,
                    subject_data,
                    self.config.sampler_seed,
                )
            print(f"[{self.TASK_NAME}] Created {len(self.subject_samplers)} samplers")
        return None
    
    @staticmethod
    def _format_subject(subject: str) -> str:
        return " ".join(subject.split("_"))
    
    @staticmethod
    def _format_example(example: Dict, include_answer: bool = True) -> str:
        letters = ["A", "B", "C", "D"]
        prompt = "Question: " + example["question"]
        for i, choice in enumerate(example["choices"]):
            prompt += f"\n{letters[i]}. {choice}"
        if include_answer:
            answer_idx = int(example["answer"])
            prompt += "\nAnswer: " + letters[answer_idx]
        else:
            prompt += "\nAnswer:"
        return prompt
    
    def process(self) -> datasets.Dataset:
        """Process with subject-specific few-shot."""
        dataset = self._load_dataset()
        if not self.config.text_only and self.config.tokenizer is None:
            raise ValueError("Tokenizer required when text_only=False")
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", range(len(dataset)))
        
        dataset = dataset.sort("subject")
        return dataset.map(
            self._format_batch_mmlu,
            batched=True,
            num_proc=16,
            remove_columns=dataset.column_names,
            desc=f"[{self.TASK_NAME}] Processing with subject-specific few-shot",
        )
    
    def _format_batch_mmlu(self, batch: Dict[str, List]) -> Dict[str, List]:
        if self.config.text_only:
            new_batch = {"context_text": [], "continuation_text": [], "is_correct": [], "group_id": [], "task_name": []}
        else:
            new_batch = {"input_ids": [], "continuation_ids": [], "is_correct": [], "group_id": [], "task_name": [], "continuation_len": [], "continuation_char_len": []}
        
        for i in range(len(batch[self.QUESTION_COL])):
            subject = batch['subject'][i]
            header = f"The following are multiple choice questions (with answers) about {self._format_subject(subject)}.\n\n"
            
            fewshot_str = ""
            if self.subject_samplers:
                sampler = self.subject_samplers.get(subject)
                if sampler:
                    samples = sampler.get_samples(self.config.num_fewshot)
                    formatted = [self._format_example(s, include_answer=True) for s in samples]
                    if formatted:
                        fewshot_str = "\n\n".join(formatted) + "\n\n"
            
            current = {"question": batch[self.QUESTION_COL][i], "choices": batch[self.OPTIONS_COL][i], "answer": batch[self.ANSWER_COL][i]}
            question_str = self._format_example(current, include_answer=False)
            prompt = header + fewshot_str + question_str
            
            group_id = batch["id"][i]
            correct_idx = int(batch[self.ANSWER_COL][i])
            options = batch[self.OPTIONS_COL][i]
            
            for j, _ in enumerate(options):
                is_correct = 1 if j == correct_idx else 0
                continuation = " " + self.CHOICE_LETTERS[j]
                if self.config.text_only:
                    new_batch["context_text"].append(prompt)
                    new_batch["continuation_text"].append(continuation)
                else:
                    self._tokenize_sample(new_batch, prompt, continuation)
                new_batch["is_correct"].append(is_correct)
                new_batch["group_id"].append(group_id)
                new_batch["task_name"].append(f"{self.TASK_NAME}:{subject}")
        return new_batch