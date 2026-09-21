# final_ema_trainer.py
import torch
import os
import contextlib
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging
import torch.distributed as dist

logger = logging.get_logger(__name__)

# --- 1. 使用更鲁棒的 ExponentialMovingAverage 类 ---
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                if s_param.device != param.device:
                    s_param.data = s_param.data.to(param.device)
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [p.clone() for p in parameters if p.requires_grad]

    def restore(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
        self.collected_params = []

    def state_dict(self):
        # 将 shadow_params 保存到 CPU，增强鲁棒性
        shadow_params_cpu = [p.cpu() for p in self.shadow_params]
        return {'decay': self.decay, 'num_updates': self.num_updates, 'shadow_params': shadow_params_cpu}

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        # 加载后，将 shadow_params 移动到正确的设备
        self.shadow_params = [p.to(device) for p in state_dict['shadow_params']]


class EMACallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        if isinstance(trainer, EMATrainer) and trainer.ema is not None:
            trainer._update_ema()


class EMATrainer(Trainer):
    def __init__(self, *args, ema_decay=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        self.ema = None
        if self.ema_decay is not None and 0 < self.ema_decay < 1:
            logger.info(f"EMA enabled with decay rate: {self.ema_decay}")
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)
            self.add_callback(EMACallback())
        else:
            logger.info("EMA is disabled.")

    def _update_ema(self):
        if self.ema is not None:
            self.ema.update(self.model.parameters())

    # --- 2. 采用你版本中更严谨的上下文管理器 ---
    @contextlib.contextmanager
    def _use_ema_for_eval(self):
        if self.ema is not None:
            logger.info("Applying EMA weights for evaluation...")
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            # 显式设置 model.eval()
            self.model.eval()
            try:
                yield
            finally:
                logger.info("Restoring original training weights after evaluation...")
                self.ema.restore(self.model.parameters())
        else:
            self.model.eval()
            try:
                yield
            finally:
                pass

    def evaluate(self, *args, **kwargs):
        with self._use_ema_for_eval():
            return super().evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with self._use_ema_for_eval():
            return super().predict(*args, **kwargs)

    # --- 3. 使用 _save_checkpoint 钩子并配合鲁棒的 EMA 类 ---
    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        if self.ema is not None and self.is_world_process_zero():
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            ema_path = os.path.join(output_dir, "ema.pt")
            logger.info(f"Saving EMA state to {ema_path}")
            torch.save(self.ema.state_dict(), ema_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.ema is not None:
            ema_path = os.path.join(resume_from_checkpoint, "ema.pt")
            if os.path.exists(ema_path):
                logger.info(f"Loading EMA state from {ema_path}")
                ema_state = torch.load(ema_path, map_location="cpu")
                self.ema.load_state_dict(ema_state, device=self.args.device)
            else:
                logger.warning(f"EMA state not found at {ema_path}. Re-initializing from model.")
                self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)
        if self.args.world_size > 1:
            dist.barrier()

    # --- 4. 增加对 save_model 的重写，确保最终保存的是 EMA 模型 ---
    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        # 首先，保存 online model，这对于恢复训练很有用
        super().save_model(output_dir, _internal_call)

        # 然后，如果启用了 EMA，用 EMA 权重覆盖并再次保存
        if self.ema is not None and self.is_world_process_zero():
            if output_dir is None:
                output_dir = self.args.output_dir
            logger.info(f"Saving EMA model for inference to {output_dir}")
            with self._use_ema_for_eval():
                # 在这个上下文中，self.model 拥有 EMA 的权重
                self.model.save_pretrained(
                    output_dir,
                    safe_serialization=self.args.save_safetensors
                )
            logger.info(f"Successfully saved EMA model to {output_dir}")