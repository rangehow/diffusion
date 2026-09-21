# sampler.py

import random
import datasets
from typing import List, Dict, Any

# --- Sampler Base Class (New Design) ---
class BaseContextSampler:
    """
    Sampler 基类，对标 lm-eval 的 ContextSampler。
    它持有一个随机数生成器，并在整个生命周期中复用。
    它的核心职责是为单个测试样本生成一个完整的 few-shot 上下文字符串。
    """
    def __init__(self, fewshot_dataset: datasets.Dataset, seed: int = 42):
        self.fewshot_dataset = fewshot_dataset
        self.rng = random.Random(seed)

    def _sample_indices(self, num_shots: int) -> List[int]:
        raise NotImplementedError("子类必须实现 _sample_indices 方法！")

    # --- 新方法：为 MMLU 等复杂任务提供灵活性 ---
    def get_samples(self, num_shots: int) -> List[Dict]:
        """
        根据采样策略，选择并返回 num_shots 个原始的 few-shot 样本。
        """
        if num_shots == 0:
            return []
        indices = self._sample_indices(num_shots)
        return self.fewshot_dataset.select(indices).to_list()

    # --- 旧方法：为 HellaSwag 等现有任务保持向后兼容 ---
    def get_context(
        self,
        num_shots: int,
        question_col: str,
        options_col: str,
        answer_col: str,
        is_sentence_completion: bool = False
    ) -> str:
        """
        为单个测试样本生成 few-shot 上下文。
        这是核心公共方法，每次被调用都会进行一次新的采样和格式化。
        """
        if num_shots == 0:
            return ""
            
        # 注意：这里我们调用 get_samples 来复用抽样逻辑
        samples = self.get_samples(num_shots)

        prompts = []
        for sample in samples:
            question = sample[question_col]
            options = sample[options_col]
            answer_idx_val = sample[answer_col]

            # (这部分逻辑与你原来的 sampler.py 完全相同)
            answer_idx = -1
            if isinstance(answer_idx_val, str):
                if answer_idx_val.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    answer_idx = ord(answer_idx_val.upper()) - ord('A')
                elif answer_idx_val.isdigit():
                    answer_idx = int(answer_idx_val)
            elif isinstance(answer_idx_val, int):
                answer_idx = answer_idx_val
            elif isinstance(answer_idx_val, list) and 1 in answer_idx_val:
                answer_idx = answer_idx_val.index(1)
            
            if 0 <= answer_idx < len(options):
                correct_answer = options[answer_idx]
                if is_sentence_completion:
                    prompt = f"{question.strip()} {correct_answer}"
                else:
                    prompt = f"Question: {question}\nAnswer: {correct_answer}"
                prompts.append(prompt)

        return "\n\n".join(prompts) + "\n\n" if prompts else ""



class FirstNSampler(BaseContextSampler):
    """总是选择数据集的前 N 个样本。"""
    def _sample_indices(self, num_shots: int) -> List[int]:
        num_to_select = min(num_shots, len(self.fewshot_dataset))
        return list(range(num_to_select))

class RandomSampler(BaseContextSampler):
    """从数据集中随机选择 N 个样本。"""
    def _sample_indices(self, num_shots: int) -> List[int]:
        num_to_select = min(num_shots, len(self.fewshot_dataset))
        return self.rng.sample(range(len(self.fewshot_dataset)), num_to_select)

class BalancedSampler(BaseContextSampler):
    """
    类别均衡采样器。
    它会尽力为每个类别选择相同数量的样本。
    注意: 这需要知道答案列 (answer_col)。
    """
    def __init__(self, fewshot_dataset: datasets.Dataset, answer_col: str, seed: int = 42):
        super().__init__(fewshot_dataset, seed)
        self.answer_col = answer_col
        self.indices_by_label = self._group_indices_by_label()

    def _group_indices_by_label(self) -> Dict[Any, List[int]]:
        indices_map = {}
        for i, item in enumerate(self.fewshot_dataset):
            label = item[self.answer_col]
            if label not in indices_map:
                indices_map[label] = []
            indices_map[label].append(i)
        return indices_map

    def _sample_indices(self, num_shots: int) -> List[int]:
        num_to_select = min(num_shots, len(self.fewshot_dataset))
        if not self.indices_by_label or num_to_select == 0:
            return []

        final_indices = []
        labels = list(self.indices_by_label.keys())
        self.rng.shuffle(labels) # 随机化类别顺序
        
        # 为本次采样创建可消耗的索引列表副本
        label_iters = {label: self.rng.sample(indices, len(indices)) for label, indices in self.indices_by_label.items()}
        
        while len(final_indices) < num_to_select:
            found_sample = False
            for label in labels:
                if label_iters[label]:
                    final_indices.append(label_iters[label].pop(0))
                    if len(final_indices) == num_to_select:
                        break
                    found_sample = True
            if not found_sample: # 如果所有类别的样本都用完了
                break
        
        return final_indices


# --- Sampler Registry ---
SAMPLER_REGISTRY = {
    "first_n": FirstNSampler,
    "random": RandomSampler,
    "balanced": BalancedSampler
}

def get_sampler(name: str, **kwargs) -> BaseContextSampler:
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"未知的采样器: {name}. 可用选项: {list(SAMPLER_REGISTRY.keys())}")
    return SAMPLER_REGISTRY[name](**kwargs)