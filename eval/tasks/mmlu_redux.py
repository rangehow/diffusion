# eval/tasks/mmlu_redux.py
"""
MMLU-Redux task implementation.

MMLU-Redux is a subset of 5,700 manually re-annotated questions across 57 MMLU subjects.
This implementation supports filtering by error type and using corrected answers.

Dataset: edinburgh-dawg/mmlu-redux-2.0
Paper: https://arxiv.org/abs/2406.04127
"""

from typing import Dict, List, Optional, Literal
import datasets
from .base import BaseTask
from .registry import register_task
from .sampler import BaseSampler, create_sampler
from ..dataset_paths import get_local_path


# Valid error types in MMLU-Redux
ERROR_TYPES = [
    "ok",                      # Correctly annotated
    "bad_question_clarity",    # Question is unclear
    "bad_options_clarity",     # Options are unclear
    "no_correct_answer",       # None of the options are correct
    "multiple_correct_answers", # More than one correct answer
    "wrong_groundtruth",       # Original MMLU label is wrong
    "expert",                  # Requires expert verification
]


@register_task("mmlu_redux")
class MMLUReduxTask(BaseTask):
    """
    MMLU-Redux evaluation task.

    By default, evaluates only on samples with error_type="ok" (verified correct).
    Can optionally include samples with corrected ground truth.

    Attributes:
        filter_mode: How to filter samples
            - "ok_only": Only use samples marked as "ok" (default)
            - "corrected": Use "ok" samples + corrected "wrong_groundtruth" samples
            - "all": Use all samples with original MMLU labels
    """

    TASK_NAME = "mmlu_redux"
    QUESTION_COL = "question"
    OPTIONS_COL = "choices"
    ANSWER_COL = "answer"
    IS_SENTENCE_COMPLETION = False

    CHOICE_LETTERS = ["A", "B", "C", "D"]

    # All 57 MMLU subjects
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

    def __init__(
        self, 
        config, 
        filter_mode: Literal["ok_only", "corrected", "all"] = "ok_only",
        use_corrected_answers: bool = True,
    ):
        """
        Initialize MMLU-Redux task.

        Args:
            config: TaskConfig instance
            filter_mode: How to filter samples:
                - "ok_only": Only verified correct samples (recommended)
                - "corrected": Include corrected wrong_groundtruth samples
                - "all": All samples with original labels
            use_corrected_answers: If True, use corrected answers for 
                wrong_groundtruth samples when filter_mode="corrected"
        """
        self.filter_mode = filter_mode
        self.use_corrected_answers = use_corrected_answers
        self.subject_samplers: Dict[str, BaseSampler] = {}
        super().__init__(config)

        print(f"[{self.TASK_NAME}] Filter mode: {filter_mode}")
        print(f"[{self.TASK_NAME}] Use corrected answers: {use_corrected_answers}")

    def _load_subject_dataset(self, subject: str) -> Optional[datasets.Dataset]:
        """
        Load a single subject's dataset with local fallback.

        Args:
            subject: MMLU subject name

        Returns:
            Dataset for the subject, or None if loading failed
        """
        # Check for local path first
        local_path = get_local_path(self.TASK_NAME)

        if local_path:
            print(f"[{self.TASK_NAME}] Trying local path for {subject}: {local_path}")
            try:
                return datasets.load_dataset(
                    local_path,
                    subject,
                    split='test',
                )
            except Exception as e:
                print(f"[{self.TASK_NAME}] Local load failed for {subject}: {e}")

        # Fallback to HuggingFace Hub
        print(f"[{self.TASK_NAME}] Loading {subject} from HuggingFace Hub...")
        try:
            return datasets.load_dataset(
                'edinburgh-dawg/mmlu-redux-2.0',
                subject,
                split='test',
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[{self.TASK_NAME}] Failed to load {subject}: {e}")
            return None

    def _load_dataset(self) -> datasets.Dataset:
        """
        Load MMLU-Redux dataset with filtering based on error_type.
        Uses local path if available, falls back to HuggingFace Hub.
        """
        all_data = []

        print(f"[{self.TASK_NAME}] Loading MMLU-Redux dataset...")

        for subject in self.SUBJECTS:
            try:
                subject_data = self._load_subject_dataset(subject)

                if subject_data is None:
                    continue

                # Add subject column if not present
                if 'subject' not in subject_data.column_names:
                    subject_data = subject_data.map(
                        lambda x: {**x, 'subject': subject},
                        desc=f"Adding subject: {subject}"
                    )

                all_data.append(subject_data)

            except Exception as e:
                print(f"[{self.TASK_NAME}] Warning: Could not load subject '{subject}': {e}")
                continue

        if not all_data:
            raise ValueError("Could not load any MMLU-Redux subjects!")

        # Concatenate all subjects
        dataset = datasets.concatenate_datasets(all_data)

        print(f"[{self.TASK_NAME}] Total samples before filtering: {len(dataset)}")

        # Apply filtering based on filter_mode
        if self.filter_mode == "ok_only":
            dataset = dataset.filter(
                lambda x: x['error_type'] == 'ok',
                desc="Filtering to 'ok' samples only"
            )
        elif self.filter_mode == "corrected":
            # Keep "ok" samples and "wrong_groundtruth" samples
            dataset = dataset.filter(
                lambda x: x['error_type'] in ['ok', 'wrong_groundtruth'],
                desc="Filtering to 'ok' and 'wrong_groundtruth' samples"
            )
        # For "all" mode, keep everything

        print(f"[{self.TASK_NAME}] Total samples after filtering: {len(dataset)}")

        # Print error type distribution
        error_counts = {}
        for item in dataset:
            et = item['error_type']
            error_counts[et] = error_counts.get(et, 0) + 1
        print(f"[{self.TASK_NAME}] Error type distribution: {error_counts}")

        return dataset

    def _load_fewshot_dataset(self) -> Optional[datasets.Dataset]:
        """
        Load few-shot examples from original MMLU dev set.
        Uses local path if available, falls back to HuggingFace Hub.

        Note: MMLU-Redux doesn't have its own dev set, so we use
        the original MMLU dev set for few-shot examples.
        """
        if self.config.num_fewshot == 0:
            return None

        print(f"[{self.TASK_NAME}] Loading MMLU dev set for few-shot examples...")

        # Check for local MMLU path
        local_path = get_local_path("mmlu")

        if local_path:
            print(f"[{self.TASK_NAME}] Trying local MMLU path: {local_path}")
            try:
                return datasets.load_dataset(
                    local_path, 
                    'all',
                    split='dev',
                )
            except Exception as e:
                print(f"[{self.TASK_NAME}] Local MMLU load failed: {e}")

        # Fallback to HuggingFace Hub
        try:
            return datasets.load_dataset(
                'cais/mmlu', 
                'all',
                split='dev',
                cache_dir=self.config.local_dir,
            )
        except Exception as e:
            print(f"[{self.TASK_NAME}] Warning: Could not load few-shot dataset: {e}")
            return None

    def _setup_sampler(self) -> Optional[BaseSampler]:
        """Set up subject-specific samplers for few-shot."""
        if self.config.num_fewshot > 0 and self.fewshot_dataset:
            print(f"[{self.TASK_NAME}] Setting up subject-specific samplers...")

            for subject in self.SUBJECTS:
                subject_data = self.fewshot_dataset.filter(
                    lambda x: x["subject"] == subject
                )

                if len(subject_data) == 0:
                    continue

                if len(subject_data) < self.config.num_fewshot:
                    print(f"  Warning: {subject} has only {len(subject_data)} few-shot examples")

                self.subject_samplers[subject] = create_sampler(
                    self.config.sampler_type,
                    subject_data,
                    self.config.sampler_seed,
                )

            print(f"[{self.TASK_NAME}] Created {len(self.subject_samplers)} subject samplers")

        return None

    @staticmethod
    def _format_subject(subject: str) -> str:
        """Format subject name for display."""
        return " ".join(subject.split("_"))

    def _format_example(self, example: Dict, include_answer: bool = True) -> str:
        """Format a single example for prompt."""
        prompt = "Question: " + example["question"]

        for i, choice in enumerate(example["choices"]):
            prompt += f"\n{self.CHOICE_LETTERS[i]}. {choice}"

        if include_answer:
            answer_idx = int(example["answer"])
            prompt += "\nAnswer: " + self.CHOICE_LETTERS[answer_idx]
        else:
            prompt += "\nAnswer:"

        return prompt

    def _get_corrected_answer(self, example: Dict) -> int:
        """
        Get the corrected answer index for a sample.

        For wrong_groundtruth samples, parse the correct_answer field.
        Returns the original answer if no correction available.
        """
        if example['error_type'] != 'wrong_groundtruth':
            return int(example['answer'])

        if not self.use_corrected_answers:
            return int(example['answer'])

        correct_answer = example.get('correct_answer', '')

        if not correct_answer:
            return int(example['answer'])

        # Try to parse the corrected answer
        correct_answer = correct_answer.strip().upper()

        # Check if it's a letter (A, B, C, D)
        if correct_answer in self.CHOICE_LETTERS:
            return self.CHOICE_LETTERS.index(correct_answer)

        # Check if it's a number (0, 1, 2, 3)
        if correct_answer.isdigit():
            idx = int(correct_answer)
            if 0 <= idx < len(example['choices']):
                return idx

        # Try to match the answer text to choices
        for i, choice in enumerate(example['choices']):
            if correct_answer.lower() == choice.lower().strip():
                return i

        # Fallback to original answer
        print(f"[{self.TASK_NAME}] Warning: Could not parse corrected answer '{correct_answer}', using original")
        return int(example['answer'])

    def process(self) -> datasets.Dataset:
        """Process the dataset into evaluation format."""
        dataset = self._load_dataset()

        if not self.config.text_only and self.config.tokenizer is None:
            raise ValueError("Tokenizer required when text_only=False")

        # Add IDs if not present
        if "id" not in dataset.column_names:
            dataset = dataset.add_column("id", range(len(dataset)))

        # Sort by subject for consistent ordering
        dataset = dataset.sort("subject")

        return dataset.map(
            self._format_batch_mmlu_redux,
            batched=True,
            num_proc=16,
            remove_columns=dataset.column_names,
            desc=f"[{self.TASK_NAME}] Processing",
            load_from_cache_file=False,
        )

    def _format_batch_mmlu_redux(self, batch: Dict[str, List]) -> Dict[str, List]:
        """Format a batch of MMLU-Redux samples."""
        if self.config.text_only:
            new_batch = {
                "context_text": [],
                "continuation_text": [],
                "is_correct": [],
                "group_id": [],
                "task_name": [],
                "error_type": [],
            }
        else:
            new_batch = {
                "input_ids": [],
                "continuation_ids": [],
                "is_correct": [],
                "group_id": [],
                "task_name": [],
                "continuation_len": [],
                "continuation_char_len": [],
                "error_type": [],
            }

        batch_size = len(batch[self.QUESTION_COL])

        for i in range(batch_size):
            subject = batch['subject'][i]
            error_type = batch['error_type'][i]

            # Build header
            header = (
                f"The following are multiple choice questions (with answers) "
                f"about {self._format_subject(subject)}.\n\n"
            )

            # Build few-shot context
            fewshot_str = ""
            if self.subject_samplers:
                sampler = self.subject_samplers.get(subject)
                if sampler:
                    samples = sampler.get_samples(self.config.num_fewshot)
                    formatted = [
                        self._format_example(s, include_answer=True) 
                        for s in samples
                    ]
                    if formatted:
                        fewshot_str = "\n\n".join(formatted) + "\n\n"

            # Build current question
            current_example = {
                "question": batch[self.QUESTION_COL][i],
                "choices": batch[self.OPTIONS_COL][i],
                "answer": batch[self.ANSWER_COL][i],
                "error_type": error_type,
                "correct_answer": batch.get('correct_answer', [''] * batch_size)[i],
            }

            question_str = self._format_example(current_example, include_answer=False)
            prompt = header + fewshot_str + question_str

            group_id = batch["id"][i]
            options = batch[self.OPTIONS_COL][i]

            # Get the correct answer index (possibly corrected)
            correct_idx = self._get_corrected_answer(current_example)

            # Create one sample per option
            for j in range(len(options)):
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
                new_batch["error_type"].append(error_type)

        return new_batch


@register_task("mmlu_redux_ok")
class MMLUReduxOkTask(MMLUReduxTask):
    """
    MMLU-Redux with only 'ok' (verified correct) samples.
    This is the recommended configuration for fair evaluation.
    """

    TASK_NAME = "mmlu_redux_ok"

    def __init__(self, config):
        super().__init__(config, filter_mode="ok_only", use_corrected_answers=False)


@register_task("mmlu_redux_corrected")
class MMLUReduxCorrectedTask(MMLUReduxTask):
    """
    MMLU-Redux with 'ok' samples plus corrected 'wrong_groundtruth' samples.
    Uses the corrected answers from the annotations.
    """

    TASK_NAME = "mmlu_redux_corrected"

    def __init__(self, config):
        super().__init__(config, filter_mode="corrected", use_corrected_answers=True)


@register_task("mmlu_redux_all")
class MMLUReduxAllTask(MMLUReduxTask):
    """
    MMLU-Redux with all samples using original MMLU labels.
    Useful for comparing against original MMLU benchmark results.
    """

    TASK_NAME = "mmlu_redux_all"

    def __init__(self, config):
        super().__init__(config, filter_mode="all", use_corrected_answers=False)