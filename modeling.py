from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from transformers import AutoModel, ModernBertConfig, ModernBertPreTrainedModel, ModernBertModel
from transformers.modeling_outputs import ModelOutput
from transformers.activations import ACT2FN
from torch import nn

from transformers.loss.loss_utils import ForMaskedLMLoss,fixed_cross_entropy
from transformers.generation import GenerationMixin

import logging  
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 


if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _unpad_modernbert_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    token_change_labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    unpadded_token_change_labels = token_change_labels.flatten()[indices] if token_change_labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels, unpadded_token_change_labels



def _pad_modernbert_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """
    Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs

def ForTokenChange(logits: torch.Tensor, labels, **kwargs):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.view(-1, 2)
    labels = labels.view(-1).to(logits.device)
    logits = logits.float()
    # Flatten the tokens
    return fixed_cross_entropy(logits, labels, **kwargs)



@dataclass
class MLMAndTokenChangeOutput(ModelOutput):
    """
    我们自定义的模型输出类
    """
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    token_change_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    token_change_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    current_mlm_prob: Optional[torch.Tensor] = None



class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))







class ModernBertForDiffusionLMWithTokenChangeHead(ModernBertPreTrainedModel,GenerationMixin):
    _tied_weights_keys = ["mlm_classifier.weight"]

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config

        # 1. 共享的 Transformer 骨干
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)

        # 稀疏预测的标志，只在训练时有用，只计算有效loss区域
        self.sparse_prediction = self.config.sparse_prediction 
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        if self.sparse_prediction:
            logger.info("This model doesn't support sparse prediction since token change task requires all token features")

        # MLM task
        self.mlm_classifier = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
        
        # Token change task
        # 将 hidden_size 映射到 2 (0: 未改变, 1: 改变)
        self.token_change_classifier = nn.Linear(config.hidden_size, 2)
        
        # (可选但推荐) 为多任务损失设置权重
        self.mlm_loss_weight = config.mlm_loss_weight if hasattr(config, "mlm_loss_weight") else 1.0
        self.token_change_loss_weight = config.token_change_loss_weight if hasattr(config, "token_change_loss_weight") else 1.0

        # 初始化权重
        self.post_init()

    def get_output_embeddings(self):
        return self.mlm_classifier

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.mlm_classifier = new_embeddings


    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.head(output)

    @torch.compile(dynamic=True)
    def compiled_mlm_classifier(self, output: torch.Tensor) -> torch.Tensor:
        return self.mlm_classifier(output)

    @torch.compile(dynamic=True)
    def compiled_change_classifier(self, output: torch.Tensor) -> torch.Tensor:
        return self.token_change_classifier(output)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_change_labels: Optional[torch.Tensor] = None,
        current_mlm_prob: Optional[torch.Tensor] = None,
        **kwargs,
    )  -> Union[Tuple[torch.Tensor], MLMAndTokenChangeOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self._maybe_set_compile()

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels,token_change_labels = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels,token_change_labels=token_change_labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels,token_change_labels = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels,token_change_labels=token_change_labels
                    )

        # 1. 通过共享的骨干网络
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state

        # 初始化损失
        total_loss = None
        mlm_loss = None
        token_change_loss = None

        # if self.sparse_prediction and labels is not None:
        #     # flatten labels and output first
        #     labels = labels.view(-1)
        #     mlm_last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

        #     # then filter out the non-masked tokens
        #     mask_tokens = labels != self.sparse_pred_ignore_index
        #     last_hidden_state = last_hidden_state[mask_tokens]
        #     labels = labels[mask_tokens]

        shared_features = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.head(last_hidden_state)
        )

        mlm_logits = (
            self.compiled_mlm_classifier(shared_features)
            if self.config.reference_compile
            else self.mlm_classifier(shared_features)
        )

        token_change_logits = (
            self.compiled_change_classifier(shared_features)
            if self.config.reference_compile
            else self.token_change_classifier(shared_features)
        )

  
        if labels is not None:
            mlm_loss = ForMaskedLMLoss(mlm_logits, labels, vocab_size=self.config.vocab_size)

        if token_change_labels is not None:
   
            token_change_loss = ForTokenChange(token_change_logits, token_change_labels)
 

        # 后处理，还原去pad的序列
        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                mlm_logits = _pad_modernbert_output(inputs=mlm_logits, indices=indices, batch=batch_size, seqlen=seq_len)
                token_change_logits = _pad_modernbert_output(inputs=token_change_logits, indices=indices, batch=batch_size, seqlen=seq_len)


        if labels is not None and  token_change_labels is not None:
            total_loss = (self.mlm_loss_weight * mlm_loss) + (self.token_change_loss_weight * token_change_loss)
            
        if not return_dict:
            # 如果不返回字典，则按元组格式输出
            # 注意：这可能会变得很复杂，推荐使用 return_dict=True
            output = (mlm_logits, token_change_logits) + outputs[1:] # outputs[1:] 包含 hidden_states, attentions
            return ((total_loss,) + output) if total_loss is not None else output

        return MLMAndTokenChangeOutput(
            loss=total_loss,
            mlm_loss=mlm_loss,
            token_change_loss=token_change_loss,
            mlm_logits=mlm_logits,
            token_change_logits=token_change_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            current_mlm_prob=current_mlm_prob,
        )


    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        mask_token_id: Optional[int],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        num_diffusion_steps: int = 10,
        temperature_mlm: float = 1.0,
        use_token_change_classifier = True,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        debug: bool = False,
        tokenizer = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        自定义的扩散生成方法
        
        Args:
            input_ids: 输入的token ids，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
            max_new_tokens: 要生成的新token数量 (L)
            num_diffusion_steps: 扩散迭代次数 (T)
            temperature_mlm: MLM采样的温度参数
            do_sample: 是否使用采样，False则使用贪心解码
            top_k: top-k采样参数
            top_p: top-p采样参数
            mask_token_id: mask token的id，如果为None则尝试自动获取
            debug: 是否启用调试模式，输出每步迭代的详细信息
            tokenizer: 用于将token id转换为文本的tokenizer（可选）
        
        Returns:
            生成的完整序列，形状为 (batch_size, original_seq_len + max_new_tokens)
        """
        batch_size, original_seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. 在输入后填充L个mask token
        mask_tokens = torch.full((batch_size, max_new_tokens), mask_token_id, 
                                dtype=input_ids.dtype, device=device)
        extended_input_ids = torch.cat([input_ids, mask_tokens], dim=1)
        
        # 扩展attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        extended_attention_mask = torch.cat([
            attention_mask, 
            torch.ones((batch_size, max_new_tokens), dtype=torch.bool, device=device)
        ], dim=1)
        
        # 2. 迭代T次
        current_sequence = extended_input_ids.clone()
        
        if debug:
            logger.info("=" * 80)
            logger.info("🚀 开始扩散生成过程")
            logger.info(f"📊 参数设置: max_new_tokens={max_new_tokens}, num_diffusion_steps={num_diffusion_steps}")
            logger.info(f"🎯 采样设置: do_sample={do_sample}, temperature_mlm={temperature_mlm}")
            if top_k is not None:
                logger.info(f"🔝 Top-k采样: k={top_k}")
            if top_p is not None:
                logger.info(f"🎲 Top-p采样: p={top_p}")
            
            # 显示初始序列
            for batch_idx in range(batch_size):
                initial_text = self._tokens_to_text(current_sequence[batch_idx], tokenizer, mask_token_id)
                logger.info(f"📝 Batch {batch_idx} 初始序列: {initial_text}")
            logger.info("=" * 80)
        
        for step in range(num_diffusion_steps):
            if debug:
                logger.info(f"\n🔄 === 步骤 {step + 1}/{num_diffusion_steps} ===")
            
            # 保存当前序列用于比较
            prev_sequence = current_sequence.clone()
            
            outputs = self.forward(
                input_ids=current_sequence,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            
            mlm_logits = outputs.mlm_logits  # (batch_size, seq_len, vocab_size)
            token_change_logits = outputs.token_change_logits  # (batch_size, seq_len, 2)

            # 只处理需要生成的部分（后L个位置）
            generation_start_idx = original_seq_len
            mlm_logits_gen = mlm_logits[:, generation_start_idx:, :]  # (batch_size, max_new_tokens, vocab_size)
            change_logits_gen = token_change_logits[:, generation_start_idx:, :]  # (batch_size, max_new_tokens, 2)
            
            # 3. 从MLM logits中生成候选token
            if do_sample:
                # 采样生成
                mlm_logits_gen = mlm_logits_gen / temperature_mlm
                
                if top_k is not None:
                    # Top-k采样
                    top_k_logits, top_k_indices = torch.topk(mlm_logits_gen, k=min(top_k, mlm_logits_gen.size(-1)))
                    mlm_logits_gen = torch.full_like(mlm_logits_gen, float('-inf'))
                    mlm_logits_gen.scatter_(-1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    # Top-p采样
                    sorted_logits, sorted_indices = torch.sort(mlm_logits_gen, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    mlm_logits_gen = mlm_logits_gen.masked_fill(indices_to_remove, float('-inf'))
                
                # 多项式采样
                probs = torch.softmax(mlm_logits_gen, dim=-1)
                candidate_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, max_new_tokens)
            else:
                # 贪心解码
                candidate_tokens = torch.argmax(mlm_logits_gen, dim=-1)  # (batch_size, max_new_tokens)
            

            change_decisions = torch.argmax(change_logits_gen, dim=-1)  # (batch_size, max_new_tokens)
            
            # 5. 应用变化
            # 只有当change_decisions为1时才更新token
            mask_positions = current_sequence[:, generation_start_idx:] == mask_token_id
            should_change = change_decisions == 1
            
            if use_token_change_classifier:
                new_tokens = torch.where(should_change, candidate_tokens, 
                                    current_sequence[:, generation_start_idx:])

                current_sequence[:, generation_start_idx:] = new_tokens
            else:
                current_sequence[:, generation_start_idx:] = candidate_tokens

            if debug:
                self._debug_step_changes(
                    step + 1,
                    prev_sequence,
                    current_sequence,
                    candidate_tokens,
                    change_decisions,
                    should_change,
                    mask_positions,
                    generation_start_idx,
                    batch_size,
                    tokenizer,
                    mask_token_id
                )

        if debug:
            logger.info("\n" + "=" * 80)
            logger.info("🎉 扩散生成完成!")
            for batch_idx in range(batch_size):
                final_text = self._tokens_to_text(current_sequence[batch_idx], tokenizer, mask_token_id)
                logger.info(f"📝 Batch {batch_idx} 最终序列: {final_text}")
            logger.info("=" * 80)
        
        return current_sequence

    def _debug_step_changes(
        self,
        step: int,
        prev_sequence: torch.Tensor,
        current_sequence: torch.Tensor,
        candidate_tokens: torch.Tensor,
        change_decisions: torch.Tensor,
        should_change: torch.Tensor,
        mask_positions: torch.Tensor,
        generation_start_idx: int,
        batch_size: int,
        tokenizer,
        mask_token_id: int
    ):
        """
        输出每步迭代的详细变化信息
        """
        # 直接通过前后序列对比找出变化
        prev_gen_tokens = prev_sequence[:, generation_start_idx:]
        curr_gen_tokens = current_sequence[:, generation_start_idx:]
        actual_changes = prev_gen_tokens != curr_gen_tokens  # (batch_size, max_new_tokens)
        
        total_changes = actual_changes.sum().item()
        total_masks = mask_positions.sum().item()
        
        logger.info(f"📈 统计信息:")
        logger.info(f"   • 剩余MASK位置: {total_masks}")
        logger.info(f"   • 实际发生的变化: {total_changes}")
        
        # 对每个batch进行详细分析
        for batch_idx in range(batch_size):
            if batch_size > 1:
                logger.info(f"\n🔍 === Batch {batch_idx} 详细分析 ===")
            
            prev_tokens = prev_gen_tokens[batch_idx]
            curr_tokens = curr_gen_tokens[batch_idx]
            candidates = candidate_tokens[batch_idx]
            changes = actual_changes[batch_idx]
            masks = mask_positions[batch_idx]
            
            # 找出所有发生变化的位置
            changed_positions = torch.where(changes)[0].tolist()
            mask_positions_list = torch.where(masks)[0].tolist()
            
            if changed_positions:
                logger.info(f"✅ 发生变化的位置 ({len(changed_positions)}个):")
                for pos in changed_positions:
                    prev_token = prev_tokens[pos].item()
                    curr_token = curr_tokens[pos].item()
                    candidate_token = candidates[pos].item()
                    
                    prev_text = self._token_to_text(prev_token, tokenizer, mask_token_id)
                    curr_text = self._token_to_text(curr_token, tokenizer, mask_token_id)
                    candidate_text = self._token_to_text(candidate_token, tokenizer, mask_token_id)
                    
                    logger.info(f"   位置 {pos:2d}: {prev_text} → {curr_text} (候选: {candidate_text})")
            else:
                logger.info("❌ 本步骤没有发生任何变化")
            
            # 显示候选token与实际选择不同的位置
            candidate_different = candidates != curr_tokens
            different_but_unchanged = candidate_different & ~changes
            different_positions = torch.where(different_but_unchanged)[0].tolist()
            
            if different_positions:
                logger.info(f"🤔 候选与实际不同但未变化的位置 ({len(different_positions)}个):")
                for pos in different_positions:
                    curr_token = curr_tokens[pos].item()
                    candidate_token = candidates[pos].item()
                    
                    curr_text = self._token_to_text(curr_token, tokenizer, mask_token_id)
                    candidate_text = self._token_to_text(candidate_token, tokenizer, mask_token_id)
                    
                    logger.info(f"   位置 {pos:2d}: 保持 {curr_text} (候选: {candidate_text})")
            
            # 显示剩余的MASK位置
            remaining_masks = torch.where(curr_tokens == mask_token_id)[0].tolist()
            if remaining_masks:
                logger.info(f"🎭 剩余MASK位置 ({len(remaining_masks)}个): {remaining_masks}")
            else:
                logger.info("🎊 所有MASK已被替换!")
            
            # 显示当前生成部分的完整文本
            current_gen_text = self._tokens_to_text(curr_tokens, tokenizer, mask_token_id)
            logger.info(f"📄 当前生成部分: {current_gen_text}")

    def _tokens_to_text(self, tokens: torch.Tensor, tokenizer, mask_token_id: int) -> str:
        """
        将token序列转换为可读文本
        """
        if tokenizer is None:
            # 如果没有tokenizer，直接显示token id
            token_strs = []
            for token_id in tokens.tolist():
                if token_id == mask_token_id:
                    token_strs.append("[MASK]")
                else:
                    token_strs.append(f"<{token_id}>")
            return " ".join(token_strs)
        else:
            # 使用tokenizer解码
            try:
                # 将MASK token替换为特殊标记以便正确显示
                display_tokens = tokens.clone()
                display_tokens[tokens == mask_token_id] = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else mask_token_id
                text = tokenizer.decode(display_tokens, skip_special_tokens=False)
                return text
            except Exception as e:
                logger.warning(f"Tokenizer解码失败: {e}")
                return self._tokens_to_text(tokens, None, mask_token_id)

    def _token_to_text(self, token_id: int, tokenizer, mask_token_id: int) -> str:
        """
        将单个token id转换为可读文本
        """
        if token_id == mask_token_id:
            return "[MASK]"
        
        if tokenizer is None:
            return f"<{token_id}>"
        else:
            try:
                text = tokenizer.decode([token_id], skip_special_tokens=False)
                return f"'{text}'"
            except Exception as e:
                return f"<{token_id}>"
            

class ModernBertForDiffusionLM(ModernBertPreTrainedModel,GenerationMixin):
    _tied_weights_keys = ["mlm_classifier.weight"]

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config

        # 1. 共享的 Transformer 骨干
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)

        # 稀疏预测的标志，只在训练时有用，只计算有效loss区域
        self.sparse_prediction = self.config.sparse_prediction 
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # MLM task
        self.mlm_classifier = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
        
        # 初始化权重
        self.post_init()

    def get_output_embeddings(self):
        return self.mlm_classifier

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.mlm_classifier = new_embeddings


    @torch.compile(dynamic=True)
    def compiled_mlm_classifier(self, output: torch.Tensor) -> torch.Tensor:
        return self.mlm_classifier(self.head(output))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_change_labels: Optional[torch.Tensor] = None,
        current_mlm_prob: Optional[torch.Tensor] = None,
        **kwargs,
    )  -> Union[Tuple[torch.Tensor], MLMAndTokenChangeOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self._maybe_set_compile()

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels,token_change_labels = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels,token_change_labels=token_change_labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels,token_change_labels = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels,token_change_labels=token_change_labels
                    )

        # 1. 通过共享的骨干网络
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state

        # 初始化损失
        total_loss = None
        mlm_loss = None
        token_change_loss = None

        # if self.sparse_prediction and labels is not None:
        #     # flatten labels and output first
        #     labels = labels.view(-1)
        #     mlm_last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

        #     # then filter out the non-masked tokens
        #     mask_tokens = labels != self.sparse_pred_ignore_index
        #     last_hidden_state = last_hidden_state[mask_tokens]
        #     labels = labels[mask_tokens]


        mlm_logits = (
            self.compiled_mlm_classifier(last_hidden_state)
            if self.config.reference_compile
            else self.mlm_classifier(self.head(last_hidden_state))
        )


        if labels is not None:
            loss = ForMaskedLMLoss(mlm_logits, labels, vocab_size=self.config.vocab_size)

        # 后处理，还原去pad的序列
        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                mlm_logits = _pad_modernbert_output(inputs=mlm_logits, indices=indices, batch=batch_size, seqlen=seq_len)
                

        
        if not return_dict:
            assert False,"还没处理"
            output = (mlm_logits, token_change_logits) + outputs[1:] # outputs[1:] 包含 hidden_states, attentions
            return ((total_loss,) + output) if total_loss is not None else output

        return MLMAndTokenChangeOutput(
            loss=loss,
            mlm_loss=loss,
            token_change_loss='NA',
            mlm_logits=mlm_logits,
            token_change_logits='NA',
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            current_mlm_prob=current_mlm_prob,
        )


    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        mask_token_id: Optional[int],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        num_diffusion_steps: int = 10,
        temperature_mlm: float = 1.0,
        use_token_change_classifier = True,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        debug: bool = False,
        tokenizer = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        自定义的扩散生成方法
        
        Args:
            input_ids: 输入的token ids，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
            max_new_tokens: 要生成的新token数量 (L)
            num_diffusion_steps: 扩散迭代次数 (T)
            temperature_mlm: MLM采样的温度参数
            do_sample: 是否使用采样，False则使用贪心解码
            top_k: top-k采样参数
            top_p: top-p采样参数
            mask_token_id: mask token的id，如果为None则尝试自动获取
            debug: 是否启用调试模式，输出每步迭代的详细信息
            tokenizer: 用于将token id转换为文本的tokenizer（可选）
        
        Returns:
            生成的完整序列，形状为 (batch_size, original_seq_len + max_new_tokens)
        """
        batch_size, original_seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. 在输入后填充L个mask token
        mask_tokens = torch.full((batch_size, max_new_tokens), mask_token_id, 
                                dtype=input_ids.dtype, device=device)
        extended_input_ids = torch.cat([input_ids, mask_tokens], dim=1)
        
        # 扩展attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        extended_attention_mask = torch.cat([
            attention_mask, 
            torch.ones((batch_size, max_new_tokens), dtype=torch.bool, device=device)
        ], dim=1)
        
        # 2. 迭代T次
        current_sequence = extended_input_ids.clone()
        
        if debug:
            logger.info("=" * 80)
            logger.info("🚀 开始扩散生成过程")
            logger.info(f"📊 参数设置: max_new_tokens={max_new_tokens}, num_diffusion_steps={num_diffusion_steps}")
            logger.info(f"🎯 采样设置: do_sample={do_sample}, temperature_mlm={temperature_mlm}")
            if top_k is not None:
                logger.info(f"🔝 Top-k采样: k={top_k}")
            if top_p is not None:
                logger.info(f"🎲 Top-p采样: p={top_p}")
            
            # 显示初始序列
            for batch_idx in range(batch_size):
                initial_text = self._tokens_to_text(current_sequence[batch_idx], tokenizer, mask_token_id)
                logger.info(f"📝 Batch {batch_idx} 初始序列: {initial_text}")
            logger.info("=" * 80)
        
        for step in range(num_diffusion_steps):
            if debug:
                logger.info(f"\n🔄 === 步骤 {step + 1}/{num_diffusion_steps} ===")
            
            # 保存当前序列用于比较
            prev_sequence = current_sequence.clone()
            
            outputs = self.forward(
                input_ids=current_sequence,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            
            mlm_logits = outputs.mlm_logits  # (batch_size, seq_len, vocab_size)
            token_change_logits = outputs.token_change_logits  # (batch_size, seq_len, 2)

            # 只处理需要生成的部分（后L个位置）
            generation_start_idx = original_seq_len
            mlm_logits_gen = mlm_logits[:, generation_start_idx:, :]  # (batch_size, max_new_tokens, vocab_size)
            change_logits_gen = token_change_logits[:, generation_start_idx:, :]  # (batch_size, max_new_tokens, 2)
            
            # 3. 从MLM logits中生成候选token
            if do_sample:
                # 采样生成
                mlm_logits_gen = mlm_logits_gen / temperature_mlm
                
                if top_k is not None:
                    # Top-k采样
                    top_k_logits, top_k_indices = torch.topk(mlm_logits_gen, k=min(top_k, mlm_logits_gen.size(-1)))
                    mlm_logits_gen = torch.full_like(mlm_logits_gen, float('-inf'))
                    mlm_logits_gen.scatter_(-1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    # Top-p采样
                    sorted_logits, sorted_indices = torch.sort(mlm_logits_gen, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    mlm_logits_gen = mlm_logits_gen.masked_fill(indices_to_remove, float('-inf'))
                
                # 多项式采样
                probs = torch.softmax(mlm_logits_gen, dim=-1)
                candidate_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, max_new_tokens)
            else:
                # 贪心解码
                candidate_tokens = torch.argmax(mlm_logits_gen, dim=-1)  # (batch_size, max_new_tokens)
            

            change_decisions = torch.argmax(change_logits_gen, dim=-1)  # (batch_size, max_new_tokens)
            
            # 5. 应用变化
            # 只有当change_decisions为1时才更新token
            mask_positions = current_sequence[:, generation_start_idx:] == mask_token_id
            should_change = change_decisions == 1
            
            if use_token_change_classifier:
                new_tokens = torch.where(should_change, candidate_tokens, 
                                    current_sequence[:, generation_start_idx:])

                current_sequence[:, generation_start_idx:] = new_tokens
            else:
                current_sequence[:, generation_start_idx:] = candidate_tokens

            if debug:
                self._debug_step_changes(
                    step + 1,
                    prev_sequence,
                    current_sequence,
                    candidate_tokens,
                    change_decisions,
                    should_change,
                    mask_positions,
                    generation_start_idx,
                    batch_size,
                    tokenizer,
                    mask_token_id
                )

        if debug:
            logger.info("\n" + "=" * 80)
            logger.info("🎉 扩散生成完成!")
            for batch_idx in range(batch_size):
                final_text = self._tokens_to_text(current_sequence[batch_idx], tokenizer, mask_token_id)
                logger.info(f"📝 Batch {batch_idx} 最终序列: {final_text}")
            logger.info("=" * 80)
        
        return current_sequence

    def _debug_step_changes(
        self,
        step: int,
        prev_sequence: torch.Tensor,
        current_sequence: torch.Tensor,
        candidate_tokens: torch.Tensor,
        change_decisions: torch.Tensor,
        should_change: torch.Tensor,
        mask_positions: torch.Tensor,
        generation_start_idx: int,
        batch_size: int,
        tokenizer,
        mask_token_id: int
    ):
        """
        输出每步迭代的详细变化信息
        """
        # 直接通过前后序列对比找出变化
        prev_gen_tokens = prev_sequence[:, generation_start_idx:]
        curr_gen_tokens = current_sequence[:, generation_start_idx:]
        actual_changes = prev_gen_tokens != curr_gen_tokens  # (batch_size, max_new_tokens)
        
        total_changes = actual_changes.sum().item()
        total_masks = mask_positions.sum().item()
        
        logger.info(f"📈 统计信息:")
        logger.info(f"   • 剩余MASK位置: {total_masks}")
        logger.info(f"   • 实际发生的变化: {total_changes}")
        
        # 对每个batch进行详细分析
        for batch_idx in range(batch_size):
            if batch_size > 1:
                logger.info(f"\n🔍 === Batch {batch_idx} 详细分析 ===")
            
            prev_tokens = prev_gen_tokens[batch_idx]
            curr_tokens = curr_gen_tokens[batch_idx]
            candidates = candidate_tokens[batch_idx]
            changes = actual_changes[batch_idx]
            masks = mask_positions[batch_idx]
            
            # 找出所有发生变化的位置
            changed_positions = torch.where(changes)[0].tolist()
            mask_positions_list = torch.where(masks)[0].tolist()
            
            if changed_positions:
                logger.info(f"✅ 发生变化的位置 ({len(changed_positions)}个):")
                for pos in changed_positions:
                    prev_token = prev_tokens[pos].item()
                    curr_token = curr_tokens[pos].item()
                    candidate_token = candidates[pos].item()
                    
                    prev_text = self._token_to_text(prev_token, tokenizer, mask_token_id)
                    curr_text = self._token_to_text(curr_token, tokenizer, mask_token_id)
                    candidate_text = self._token_to_text(candidate_token, tokenizer, mask_token_id)
                    
                    logger.info(f"   位置 {pos:2d}: {prev_text} → {curr_text} (候选: {candidate_text})")
            else:
                logger.info("❌ 本步骤没有发生任何变化")
            
            # 显示候选token与实际选择不同的位置
            candidate_different = candidates != curr_tokens
            different_but_unchanged = candidate_different & ~changes
            different_positions = torch.where(different_but_unchanged)[0].tolist()
            
            if different_positions:
                logger.info(f"🤔 候选与实际不同但未变化的位置 ({len(different_positions)}个):")
                for pos in different_positions:
                    curr_token = curr_tokens[pos].item()
                    candidate_token = candidates[pos].item()
                    
                    curr_text = self._token_to_text(curr_token, tokenizer, mask_token_id)
                    candidate_text = self._token_to_text(candidate_token, tokenizer, mask_token_id)
                    
                    logger.info(f"   位置 {pos:2d}: 保持 {curr_text} (候选: {candidate_text})")
            
            # 显示剩余的MASK位置
            remaining_masks = torch.where(curr_tokens == mask_token_id)[0].tolist()
            if remaining_masks:
                logger.info(f"🎭 剩余MASK位置 ({len(remaining_masks)}个): {remaining_masks}")
            else:
                logger.info("🎊 所有MASK已被替换!")
            
            # 显示当前生成部分的完整文本
            current_gen_text = self._tokens_to_text(curr_tokens, tokenizer, mask_token_id)
            logger.info(f"📄 当前生成部分: {current_gen_text}")

    def _tokens_to_text(self, tokens: torch.Tensor, tokenizer, mask_token_id: int) -> str:
        """
        将token序列转换为可读文本
        """
        if tokenizer is None:
            # 如果没有tokenizer，直接显示token id
            token_strs = []
            for token_id in tokens.tolist():
                if token_id == mask_token_id:
                    token_strs.append("[MASK]")
                else:
                    token_strs.append(f"<{token_id}>")
            return " ".join(token_strs)
        else:
            # 使用tokenizer解码
            try:
                # 将MASK token替换为特殊标记以便正确显示
                display_tokens = tokens.clone()
                display_tokens[tokens == mask_token_id] = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else mask_token_id
                text = tokenizer.decode(display_tokens, skip_special_tokens=False)
                return text
            except Exception as e:
                logger.warning(f"Tokenizer解码失败: {e}")
                return self._tokens_to_text(tokens, None, mask_token_id)

    def _token_to_text(self, token_id: int, tokenizer, mask_token_id: int) -> str:
        """
        将单个token id转换为可读文本
        """
        if token_id == mask_token_id:
            return "[MASK]"
        
        if tokenizer is None:
            return f"<{token_id}>"
        else:
            try:
                text = tokenizer.decode([token_id], skip_special_tokens=False)
                return f"'{text}'"
            except Exception as e:
                return f"<{token_id}>"
            
