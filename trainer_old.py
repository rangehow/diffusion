

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
# ruff: isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
)

# ruff: isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
    set_rng_state_for_device,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    check_torch_load_is_safe,
    find_labels,
    is_accelerate_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_optimi_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.import_utils import requires
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        DataLoaderConfiguration,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        release_memory,
        save_fsdp_model,
        save_fsdp_optimizer,
    )
    from accelerate.utils.memory import clear_device_cache

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,)
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    if "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"



from transformers import Trainer

class MultipleLossTrainer(Trainer):

    def __init__(self, keys_you_want_to_log=[] ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys_you_want_to_log = keys_you_want_to_log


    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        重载 create_scheduler 方法。

        检查 self.args.lr_scheduler_kwargs 中是否存在 'num_training_steps' 或 'num_warmup_steps'。
        如果存在，则使用 kwargs 中的值来创建 scheduler，否则沿用默认行为。
        这使得 scheduler 的衰减周期可以独立于总的训练步数。
        """
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        # 创建一个可修改的副本，避免影响原始配置
        lr_scheduler_kwargs = (self.args.lr_scheduler_kwargs or {}).copy()

        # --- 处理 num_training_steps ---
        scheduler_training_steps = lr_scheduler_kwargs.get("num_training_steps", num_training_steps)
        if "num_training_steps" in lr_scheduler_kwargs:
            logger.info(
                f"Overriding `num_training_steps` for scheduler. "
                f"Trainer's steps: {num_training_steps}, Scheduler's steps: {scheduler_training_steps}."
            )
            # 从字典中移除，防止重复传入
            lr_scheduler_kwargs.pop("num_training_steps")

        # --- 处理 num_warmup_steps ---
        if "num_warmup_steps" in lr_scheduler_kwargs:
            scheduler_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"]
            logger.info(
                f"Overriding `num_warmup_steps` for scheduler. "
                f"Using value from lr_scheduler_kwargs: {scheduler_warmup_steps}."
            )
            # 从字典中移除，防止重复传入
            lr_scheduler_kwargs.pop("num_warmup_steps")
        else:
            # 沿用默认逻辑，但基于 scheduler 的总步数计算 warmup
            scheduler_warmup_steps = self.args.get_warmup_steps(scheduler_training_steps)
            
        # 创建 scheduler
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=scheduler_warmup_steps,
            num_training_steps=scheduler_training_steps,
            # 将清理后的、剩余的特定参数传入
            scheduler_specific_kwargs=lr_scheduler_kwargs,
        )
        self._created_lr_scheduler = True
        return self.lr_scheduler


    # def compute_loss(
    #     self,
    #     model: nn.Module,
    #     inputs: dict[str, Union[torch.Tensor, Any]],
    #     return_outputs: bool = False,
    #     num_items_in_batch: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Args:
    #         model (`nn.Module`):
    #             The model to compute the loss for.
    #         inputs (`dict[str, Union[torch.Tensor, Any]]`):
    #             The input data for the model.
    #         return_outputs (`bool`, *optional*, defaults to `False`):
    #             Whether to return the model outputs along with the loss.
    #         num_items_in_batch (Optional[torch.Tensor], *optional*):
    #             The number of items in the batch. If num_items_in_batch is not passed,

    #     Returns:
    #         The loss of the model along with its output if return_outputs was set to True

    #     Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
    #     make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculating might be slightly inaccurate when performing gradient accumulation.
    #     """
    #     if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
    #     if self.model_accepts_loss_kwargs:
    #         kwargs = {}
    #         if num_items_in_batch is not None:
    #             kwargs["num_items_in_batch"] = num_items_in_batch
    #         inputs = {**inputs, **kwargs}
    #     outputs = model(**inputs)

    #     # User-defined compute_loss function
    #     if self.compute_loss_func is not None:
    #         if labels is None:
    #             logger.warning(
    #                 "Trainer: `compute_loss_func` is defined but `labels=None`. "
    #                 "Your custom loss function will still be called with labels=None. "
    #             )
    #         loss = self.compute_loss_func(
    #             outputs,
    #             labels,
    #             num_items_in_batch=num_items_in_batch,
    #         )
    #     # Default HF loss handling (label smoothing) if no custom loss function
    #     elif labels is not None:
    #         unwrapped_model = self.accelerator.unwrap_model(model)
    #         model_name = (
    #             unwrapped_model.base_model.model._get_name()
    #             if _is_peft_model(unwrapped_model)
    #             else unwrapped_model._get_name()
    #         )
    #         if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #             loss = self.label_smoother(outputs, labels, shift_labels=True)
    #         else:
    #             loss = self.label_smoother(outputs, labels)
    #     else:
    #         if isinstance(outputs, dict) and "loss" not in outputs:
    #             raise ValueError(
    #                 "The model did not return a loss from the inputs, only the following keys: "
    #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #             )
    #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     if (
    #         self.args.average_tokens_across_devices
    #         and (self.model_accepts_loss_kwargs or self.compute_loss_func)
    #         and num_items_in_batch is not None
    #     ):
    #         loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
    #         if isinstance(outputs, dict):
    #             for key, value in outputs.items():
    #                 if "loss" in key.lower() and key != "loss" and isinstance(value, torch.Tensor):
    #                     outputs[key] = value * self.accelerator.num_processes
                    
    #     return (loss, outputs) if return_outputs else loss


    # def training_step(
    #     self,
    #     model: nn.Module,
    #     inputs: dict[str, Union[torch.Tensor, Any]],
    #     num_items_in_batch: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.

    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     # Prepare buffers for context parallelism

    #     cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

    #     # Context manager is no-op if CP isn't enabled
    #     with cp_context():
    #         model.train()
    #         if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #             self.optimizer.train()

    #         inputs = self._prepare_inputs(inputs)
    #         if is_sagemaker_mp_enabled():
    #             loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #             return loss_mb.reduce_mean().detach().to(self.args.device)

    #         with self.compute_loss_context_manager():
    #             loss,outputs = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, return_outputs = True)
            
    #         # 提取额外的key值但不直接保存到state
    #         extra_outputs = {}
    #         if isinstance(outputs, dict):
    #             for key in self.keys_you_want_to_log:
    #                 if key in outputs and isinstance(outputs[key], torch.Tensor):
    #                     if self.args.n_gpu > 1:
    #                         outputs[key] = outputs[key].mean()
    #                     extra_outputs[key] = outputs[key].detach()

    #         # 暂存到training_step的返回值中
    #         self._current_step_extra_outputs = extra_outputs

    #         del inputs
    #         if (
    #             self.args.torch_empty_cache_steps is not None
    #             and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #         ):
    #             clear_device_cache()

    #         kwargs = {}

    #         # For LOMO optimizers you need to explicitly use the learning rate
    #         if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #             kwargs["learning_rate"] = self._get_learning_rate()

    #         if self.args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #         # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
    #         if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
    #             # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
    #             loss = loss / self.current_gradient_accumulation_steps

    #         # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
    #         # https://github.com/huggingface/transformers/pull/35808
    #         if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
    #             kwargs["scale_wrt_gas"] = False

    #         self.accelerator.backward(loss, **kwargs)

    #         return loss.detach()


    


    # def _inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     self.accelerator.free_memory()
    #     self._train_batch_size = batch_size
    #     if self.args.auto_find_batch_size:
    #         if self.state.train_batch_size != self._train_batch_size:
    #             release_memory(self.model_wrapped)
    #             self.model_wrapped = self.model

    #             # Check for DeepSpeed *after* the initial pass and modify the config
    #             if self.is_deepspeed_enabled:
    #                 # Temporarily unset `self.args.train_batch_size`
    #                 original_bs = self.args.per_device_train_batch_size
    #                 self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
    #                 self.propagate_args_to_deepspeed(True)
    #                 self.args.per_device_train_batch_size = original_bs
    #         self.state.train_batch_size = self._train_batch_size
    #     logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()
    #     if self.is_fsdp_xla_v2_enabled:
    #         train_dataloader = tpu_spmd_dataloader(train_dataloader)

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = self.get_total_train_batch_size(args)

    #     (
    #         num_train_epochs,
    #         num_update_steps_per_epoch,
    #         num_examples,
    #         num_train_samples,
    #         epoch_based,
    #         len_dataloader,
    #         max_steps,
    #     ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torchrun or torch.distributed.launch (deprecated))."
    #             )
    #         else:
    #             DebugUnderflowOverflow(self.model)

    #     delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

    #     # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
    #     is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
    #     if is_fsdp2:
    #         delay_optimizer_creation = False

    #     # We need to reset the scheduler, as its parameters may be different on subsequent calls
    #     if self._created_lr_scheduler:
    #         self.lr_scheduler = None
    #         self._created_lr_scheduler = False

    #     if self.is_deepspeed_enabled:
    #         self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

    #     if not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState(
    #         stateful_callbacks=[
    #             cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
    #         ]
    #     )
    #     self.state.is_hyper_param_search = trial is not None
    #     self.state.train_batch_size = self._train_batch_size

    #     # Compute absolute values for logging, eval, and save if given as ratio
    #     self.state.compute_steps(args, max_steps)

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

    #     model = self._wrap_model(self.model_wrapped)

    #     # as the model is wrapped, don't use `accelerator.prepare`
    #     # this is for unhandled cases such as
    #     # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    #     use_accelerator_prepare = model is self.model

    #     if use_accelerator_prepare and self.is_fsdp_enabled:
    #         # In case of auto_find_batch_size=True
    #         # Remove FSDP wrapping from sub-models.
    #         self.model = unwrap_model(self.model, recursive=True)

    #     if delay_optimizer_creation:
    #         if use_accelerator_prepare:
    #             # configure fsdp plugin for qlora if any
    #             self._fsdp_qlora_plugin_updates()
    #             if self.accelerator.mixed_precision != "fp8":
    #                 self.model = self.accelerator.prepare(self.model)
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # prepare using `accelerator` prepare
    #     if use_accelerator_prepare:
    #         self.model.train()
    #         if hasattr(self.lr_scheduler, "step"):
    #             # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
    #             if self.is_tp_enabled:
    #                 self.optimizer = self.accelerator.prepare(self.optimizer)
    #             else:
    #                 model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
    #         else:
    #             # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
    #             model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
    #                 self.model, self.optimizer, self.lr_scheduler
    #             )
    #     else:
    #         self.optimizer = self.accelerator.prepare(self.optimizer)

    #     if self.is_fsdp_enabled:
    #         self.model = self.model_wrapped = model

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     # backward compatibility
    #     if self.is_deepspeed_enabled:
    #         self.deepspeed = self.model_wrapped

    #     # ckpt loading
    #     if resume_from_checkpoint is not None:
    #         if self.is_deepspeed_enabled:
    #             deepspeed_load_checkpoint(
    #                 self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
    #             )
    #         elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
    #             self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)
    #     self._load_scaler(resume_from_checkpoint)

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    #     # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples:,}")
    #     logger.info(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    #     if self.args.per_device_train_batch_size != self._train_batch_size:
    #         logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps:,}")
    #     logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     self.initial_num_input_tokens_seen_for_session = self.state.num_input_tokens_seen
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         self.compare_trainer_and_checkpoint_args(self.args, self.state)
    #         self._load_callback_state()
    #         epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first"
    #                 f" {steps_trained_in_current_epoch} batches in the first epoch."
    #             )

    #     # Update the references
    #     for attr in ("model", "optimizer", "lr_scheduler"):
    #         setattr(self.callback_handler, attr, getattr(self, attr))
    #     self.callback_handler.train_dataloader = train_dataloader

    #     self.state.init_training_references(self, max_steps, num_train_epochs, trial)

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0, device=args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
        
    #     # 添加额外key的累积tensor
    #     tr_extra_scalars = {}
    #     for key in self.keys_you_want_to_log:
    #         tr_extra_scalars[key] = torch.tensor(0.0, device=args.device)
        
    #     model.zero_grad()
    #     grad_norm: Optional[float] = None
    #     learning_rate = None
    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    #     if args.eval_on_start:
    #         self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    #     for epoch in range(epochs_trained, num_train_epochs):
    #         epoch_dataloader = train_dataloader
    #         if hasattr(epoch_dataloader, "set_epoch"):
    #             epoch_dataloader.set_epoch(epoch)

    #         steps_in_epoch = (
    #             len(epoch_dataloader)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

    #         # ================= START OF THE FIX =================

    #         # This logic block is carefully designed to handle counter and dataloader state
    #         # for both new epochs and resumed epochs.

    #         step = -1
    #         rng_to_sync = False

    #         # Handle resumption from checkpoint
    #         if epoch == epochs_trained and resume_from_checkpoint is not None:
    #             if steps_trained_in_current_epoch > 0 and not args.ignore_data_skip:
    #                 epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
    #                 step = steps_trained_in_current_epoch - 1
    #                 rng_to_sync = True
    #             elif steps_trained_in_current_epoch == 0:
    #                 self._load_rng_state(resume_from_checkpoint)

    #         epoch_iterator = iter(epoch_dataloader)
    #         # We chunkify the epoch iterator into gradient accumulation steps `n` batches
    #         remainder = steps_in_epoch % args.gradient_accumulation_steps
    #         if remainder == 0:
    #             remainder = args.gradient_accumulation_steps
    #         update_step = -1
    #         total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
    #             remainder < args.gradient_accumulation_steps
    #         )
    #         for _ in range(total_updates):
    #             update_step += 1
    #             num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
    #             batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
    #             # Store the number of batches for current gradient accumulation
    #             # This is used to correctly scale the loss when the last accumulation step has fewer batches
    #             self.current_gradient_accumulation_steps = len(batch_samples)
    #             for i, inputs in enumerate(batch_samples):
    #                 step += 1
    #                 do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
    #                 # Since we perform prefetching, we need to manually set sync_gradients
    #                 self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

    #                 if self.args.include_num_input_tokens_seen != "no":
    #                     main_input_name = getattr(self.model, "main_input_name", "input_ids")
    #                     if main_input_name not in inputs:
    #                         logger.warning(
    #                             "Tried to track the number of tokens seen, however the current model is "
    #                             "not configured properly to know what item is the input. To fix this, add "
    #                             "a `main_input_name` attribute to the model class you are using."
    #                         )
    #                     else:
    #                         if self.args.include_num_input_tokens_seen == "non_padding":
    #                             if "attention_mask" in inputs:
    #                                 input_tokens = inputs["attention_mask"].sum()
    #                             elif (
    #                                 self.processing_class is not None
    #                                 and hasattr(self.processing_class, "pad_token_id")
    #                                 and self.processing_class.pad_token_id is not None
    #                             ):
    #                                 input_tokens = (
    #                                     inputs[main_input_name] != self.processing_class.pad_token_id
    #                                 ).sum()
    #                             else:
    #                                 logger.warning(
    #                                     "Could not determine method to count non-padding tokens, falling back to counting all tokens."
    #                                 )
    #                                 input_tokens = inputs[main_input_name].numel()
    #                         else:
    #                             input_tokens = inputs[main_input_name].numel()

    #                         input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
    #                         self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()

    #                 if rng_to_sync:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                     rng_to_sync = False

    #                 if step % args.gradient_accumulation_steps == 0:
    #                     self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

    #                 # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
    #                 context = (
    #                     functools.partial(self.accelerator.no_sync, model=model)
    #                     if i != len(batch_samples) - 1
    #                     and self.accelerator.distributed_type != DistributedType.DEEPSPEED
    #                     else contextlib.nullcontext
    #                 )
    #                 with context():
    #                     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

    #                 if hasattr(self, '_current_step_extra_outputs'):
    #                     for key, value in self._current_step_extra_outputs.items():
    #                         if key in tr_extra_scalars:
    #                             tr_extra_scalars[key] += value

    #                 if (
    #                     args.logging_nan_inf_filter
    #                     and not is_torch_xla_available()
    #                     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #                 ):
    #                     # if loss is nan or inf simply add the average of previous logged losses
    #                     tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        
    #                 else:
    #                     if tr_loss.device != tr_loss_step.device:
    #                         raise ValueError(
    #                             f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
    #                         )
    #                     tr_loss = tr_loss + tr_loss_step


    #                 self.current_flos += float(self.floating_point_ops(inputs))

    #                 if do_sync_step:
    #                     # Since we perform prefetching, we need to manually set sync_gradients to True
    #                     self.accelerator.gradient_state._set_sync_gradients(True)

    #                     # Gradient clipping
    #                     if args.max_grad_norm is not None and args.max_grad_norm > 0:
    #                         if is_sagemaker_mp_enabled() and args.fp16:
    #                             _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
    #                         else:
    #                             grad_norm_context = contextlib.nullcontext
    #                             if self.is_tp_enabled:
    #                                 from torch.distributed._tensor.experimental import implicit_replication

    #                                 grad_norm_context = implicit_replication
    #                             with grad_norm_context():
    #                                 _grad_norm = self.accelerator.clip_grad_norm_(
    #                                     model.parameters(),
    #                                     args.max_grad_norm,
    #                                 )

    #                         if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
    #                             grad_norm = model.get_global_grad_norm()
    #                             # In some cases the grad norm may not return a float
    #                             if hasattr(grad_norm, "item"):
    #                                 grad_norm = grad_norm.item()
    #                         else:
    #                             grad_norm = _grad_norm

    #                     self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

    #                     context = contextlib.nullcontext
    #                     if self.is_tp_enabled:
    #                         from torch.distributed._tensor.experimental import implicit_replication

    #                         context = implicit_replication

    #                     with context():
    #                         self.optimizer.step()

    #                     self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

    #                     # get leaning rate before update
    #                     learning_rate = self._get_learning_rate()

    #                     if not self.accelerator.optimizer_step_was_skipped:
    #                         # Delay optimizer scheduling until metrics are generated
    #                         if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #                             self.lr_scheduler.step()

    #                     model.zero_grad()
    #                     self.state.global_step += 1
    #                     self.state.epoch = epoch + (step + 1) / steps_in_epoch
    #                     self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    #                     self._maybe_log_save_evaluate(
    #                         tr_loss,
    #                         tr_extra_scalars,  # 传递额外的scalars
    #                         grad_norm,
    #                         model,
    #                         trial,
    #                         epoch,
    #                         ignore_keys_for_eval,
    #                         start_time,
    #                         learning_rate=learning_rate,
    #                     )
    #                 else:
    #                     self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

    #                 # PyTorch/XLA relies on the data loader to insert the mark_step for
    #                 # each step. Since we are breaking the loop early, we need to manually
    #                 # insert the mark_step here.
    #                 if self.control.should_epoch_stop or self.control.should_training_stop:
    #                     if is_torch_xla_available():
    #                         xm.mark_step()
    #                     break
    #             # We also need to break out of the nested loop
    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 if is_torch_xla_available():
    #                     xm.mark_step()
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems not to be a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(
    #             tr_loss, tr_extra_scalars, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
    #         )

    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_xla_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    #     train_loss = self._total_loss_scalar / effective_global_step

    #     metrics = speed_metrics(
    #         "train",
    #         start_time,
    #         num_samples=num_train_samples,
    #         num_steps=self.state.max_steps,
    #     )
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
    #                 logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint, ignore_errors=True)

    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    #     # Wait for the checkpoint to be uploaded.
    #     self._finish_current_push()

    #     # After training we make sure to retrieve back the original forward pass method
    #     # for the embedding layer by removing the forward post hook.
    #     if self.neftune_noise_alpha is not None:
    #         self._deactivate_neftune(self.model)

    #     return TrainOutput(self.state.global_step, train_loss, metrics)
    

    # def _maybe_log_save_evaluate(
    #     self, tr_loss, tr_extra_scalars, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    # ):
    #     if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
    #         if is_torch_xla_available():
    #             xm.mark_step()

    #         logs: dict[str, float] = {}

    #         # all_gather + mean() to get average loss over all processes
    #         tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

    #         # reset tr_loss to zero
    #         tr_loss -= tr_loss

    #         logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            
    #         # 处理额外的keys
    #         for key in self.keys_you_want_to_log:
    #             if key in tr_extra_scalars:
    #                 # 计算平均值并添加到logs
    #                 extra_scalar = self._nested_gather(tr_extra_scalars[key]).mean().item()
    #                 logs[key] = round(extra_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
    #                 # 重置累积tensor
    #                 tr_extra_scalars[key] -= tr_extra_scalars[key]
            
    #         if grad_norm is not None:
    #             logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    #         if learning_rate is not None:
    #             logs["learning_rate"] = learning_rate
    #         else:
    #             logs["learning_rate"] = self._get_learning_rate()

    #         self._total_loss_scalar += tr_loss_scalar
    #         self._globalstep_last_logged = self.state.global_step
    #         self.store_flos()

    #         self.log(logs, start_time)

    #     metrics = None
    #     if self.control.should_evaluate:
    #         metrics = self._evaluate(trial, ignore_keys_for_eval)
    #         is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

    #         if self.args.save_strategy == SaveStrategy.BEST:
    #             self.control.should_save = is_new_best_metric

    #     if self.control.should_save:
    #         self._save_checkpoint(model, trial)
    #         self.control = self.callback_handler.on_save(self.args, self.state, self.control)