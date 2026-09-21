import os
import torch
from contextlib import contextmanager,nullcontext
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import get_parameter_names

# ==========================================
# Part 1: Modified ExponentialMovingAverage
# 基于 mdlm 的实现，增加了 device 管理
# ==========================================

class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the result of
                `model.parameters()`.
            decay: The exponential decay.
            use_num_updates: Whether to use number of updates when computing
                averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        
        # 立即 detach 并 clone 参数，确保它们不占用计算图
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def to(self, device):
        self.shadow_params = [p.to(device) for p in self.shadow_params]
        if self.collected_params:
            self.collected_params = [p.to(device) for p in self.collected_params]

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                # s_param = s_param * decay + param * (1 - decay)
                # 变换形式以利用原地操作 sub_
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        if not self.collected_params:
            raise RuntimeError("No collected params to restore")
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        self.collected_params = []

    def state_dict(self):
        return dict(decay=self.decay,
                    num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


# ==========================================
# Part 2: EMA Callback
# 替代原本错误的 optimizer_step 覆盖
# ==========================================

class EMACallback(TrainerCallback):
    """
    HF Trainer Callback to trigger EMA update after optimizer step.
    """
    def __init__(self, trainer):
        self.trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        """
        在 optimizer.step() 之后被调用。
        """
        # 确保 EMA 已初始化
        if self.trainer.ema is None:
            self.trainer._init_ema_if_needed()
        
        # 执行更新
        self.trainer.ema.update(self.trainer._get_model_param_iter())


# ==========================================
# Part 3: Fixed EMATrainer
# ==========================================

class EMATrainer(Trainer):
    def __init__(self, ema_decay=0.9999,eval_data_collator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. 处理参数：把 0 或负数 转为 None
        if ema_decay is not None and ema_decay <= 0:
            ema_decay = None
            
        self.ema_decay = ema_decay
        self.ema = None
        self.eval_data_collator = eval_data_collator if eval_data_collator else self.data_collator

        # 注册 EMA 回调函数
        if self.ema_decay is not None:
            self.add_callback(EMACallback(self))
            if self.is_world_process_zero():
                print(f"[EMATrainer] EMA enabled with decay {self.ema_decay}")
        else:
            if self.is_world_process_zero():
                print("[EMATrainer] EMA disabled.")


    def get_eval_dataloader(self, eval_dataset=None):
        """
        重写评估 DataLoader 的获取逻辑
        """
        # 暂时保存训练用的 collator
        train_collator = self.data_collator
        
        # 偷梁换柱：将 collator 临时替换为 eval_data_collator
        self.data_collator = self.eval_data_collator
        
        try:
            # 调用父类方法创建 DataLoader，此时父类会使用 self.eval_data_collator
            dataloader = super().get_eval_dataloader(eval_dataset)
        finally:
            # 恢复训练用的 collator，以免影响后续训练步骤
            self.data_collator = train_collator
            
        return dataloader


    def _get_model_param_iter(self):
        """
        获取模型参数迭代器。
        注意：self.model 通常是原始模型，self.model_wrapped 可能是 DDP/FSDP 包装后的。
        EMA 通常跟踪 trainable parameters。
        """
        # 优先使用 unwrapped model 以避免 DDP 前缀问题，
        # 只要保证 update 和 copy_to 使用相同的顺序即可。
        return self.model.parameters()

    def _init_ema_if_needed(self):
        if self.ema is None:
            # 初始化 EMA
            self.ema = ExponentialMovingAverage(
                self._get_model_param_iter(),
                decay=self.ema_decay
            )
            # 移动到模型所在设备
            self.ema.to(self.args.device)
            
            if self.is_world_process_zero():
                print(f"[EMATrainer] EMA initialized with decay {self.ema_decay}")

    @contextmanager
    def _real_ema_context(self):
        """
        上下文管理器：进入时替换权重，退出时恢复权重。
        """
        self._init_ema_if_needed()
        
        if self.is_world_process_zero():
            print("[EMATrainer] Swapping model weights with EMA weights...")
            
        self.ema.store(self._get_model_param_iter())
        self.ema.copy_to(self._get_model_param_iter())
        try:
            yield
        finally:
            if self.is_world_process_zero():
                print("[EMATrainer] Restoring original model weights...")
            self.ema.restore(self._get_model_param_iter())

    def ema_context(self):
        """
        分发器：
        - 如果没开启 EMA，直接返回空上下文 (0 开销)
        - 如果开启了，返回真正的上下文管理器
        """
        if self.ema_decay is None:
            return nullcontext()
        
        return self._real_ema_context()

    # ------------------------------------------------------------------
    # 覆盖 evaluate 和 predict，使用上下文管理器
    # ------------------------------------------------------------------

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        with self.ema_context():
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        with self.ema_context():
            return super().predict(test_dataset, ignore_keys, metric_key_prefix)

    # ------------------------------------------------------------------
    # 覆盖 Save / Load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model, trial):
        # 调用父类保存常规 checkpoint
        super()._save_checkpoint(model, trial)
        
        # 保存 EMA 状态
        if self.ema is not None:
            checkpoint_folder = f"{self.state.best_model_checkpoint}" if self.state.best_model_checkpoint else f"checkpoint-{self.state.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            
            # 仅主进程保存
            if self.args.should_save:
                ema_path = os.path.join(output_dir, "ema_state.pt")
                torch.save(self.ema.state_dict(), ema_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # 调用父类加载常规状态
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        
        # 加载 EMA 状态
        ema_path = os.path.join(resume_from_checkpoint, "ema_state.pt")
        if os.path.exists(ema_path):
            if self.ema is None:
                # 必须先初始化结构
                self._init_ema_if_needed()
            
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            ema_state = torch.load(ema_path, map_location=map_location)
            self.ema.load_state_dict(ema_state)
            self.ema.to(self.args.device)
            
            if self.is_world_process_zero():
                print(f"[EMATrainer] Loaded EMA state from {ema_path}")
        else:
            if self.is_world_process_zero():
                print("[EMATrainer] No EMA state found in checkpoint.")