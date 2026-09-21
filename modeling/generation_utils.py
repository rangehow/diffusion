import torch
from transformers.utils import logging
from typing import Dict, Optional, Tuple, Union

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

class GenerationMixin:

    def _generate_parallel(
        self,
        input_ids: torch.LongTensor,
        mask_token_id: Optional[int],
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        num_diffusion_steps: int,
        temperature: float,
        use_token_change_classifier: bool,
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        debug: bool,
        tokenizer,
        decode_top_k_positions: Optional[int],
        force_ar_progression: bool, # 【新增】接收新参数
        **kwargs,
    ) -> torch.LongTensor:
        """原始的并行解码逻辑，一次性生成所有max_new_tokens。"""
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
        
        current_sequence = extended_input_ids.clone()

        if debug:
            logger.info("=" * 80)
            logger.info("🚀 开始并行扩散生成过程")
            logger.info(f"📊 参数设置: max_new_tokens={max_new_tokens}, num_diffusion_steps={num_diffusion_steps}")
            logger.info(f"🎯 采样设置: do_sample={do_sample}, temperature={temperature}")
            # ... (debug info)
            
        for step in range(num_diffusion_steps):
            if debug: logger.info(f"\n🔄 === 步骤 {step + 1}/{num_diffusion_steps} ===")
            prev_sequence = current_sequence.clone()

            outputs = self.forward(
                input_ids=current_sequence,
                attention_mask=extended_attention_mask,
                return_dict=True,
                causal=True
            )
            mlm_logits = outputs.logits
            generation_start_idx = original_seq_len
            
            if mlm_logits.shape[1] > generation_start_idx:
                mlm_logits_gen = mlm_logits[:, generation_start_idx - 1:-1, :]
            else:
                mlm_logits_gen = mlm_logits[:, -1, :].unsqueeze(1)
            
            assert mlm_logits_gen.shape[1] == max_new_tokens, f"Logits slice shape mismatch. Expected {max_new_tokens}, got {mlm_logits_gen.shape[1]}"

            # 3. 从MLM logits中生成候选token
            if do_sample:
                mlm_logits_gen = mlm_logits_gen / temperature
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(mlm_logits_gen, k=min(top_k, mlm_logits_gen.size(-1)))
                    mlm_logits_gen = torch.full_like(mlm_logits_gen, float('-inf'))
                    mlm_logits_gen.scatter_(-1, top_k_indices, top_k_logits)
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(mlm_logits_gen, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    mlm_logits_gen = mlm_logits_gen.masked_fill(indices_to_remove, float('-inf'))
                probs = torch.softmax(mlm_logits_gen, dim=-1)
                candidate_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, max_new_tokens)
            else:
                candidate_tokens = torch.argmax(mlm_logits_gen, dim=-1)

            # 更新策略
            current_gen_part = current_sequence[:, generation_start_idx:]
            base_update_mask = None # 用于存储基础的更新掩码
            is_mask_mask = (current_gen_part == mask_token_id)

            if decode_top_k_positions is not None:

                if not is_mask_mask.any():
                    if debug: logger.info("  [信息] 没有剩余的MASK token，跳过更新。")
                    break
                probs = torch.softmax(mlm_logits_gen, dim=-1)
                confidence_scores, _ = torch.max(probs, dim=-1)
                masked_confidence_scores = torch.where(is_mask_mask, confidence_scores, -1.0)
                k = min(decode_top_k_positions, max_new_tokens)
                _, top_k_indices = torch.topk(masked_confidence_scores, k=k, dim=1)
                base_update_mask = torch.zeros_like(confidence_scores, dtype=torch.bool, device=device)
                base_update_mask.scatter_(1, top_k_indices, True)
                base_update_mask = base_update_mask & is_mask_mask
            elif use_token_change_classifier:
                # ... (token change classifier logic) ...
                # ... (此处省略以保持简洁) ...
                base_update_mask = ... # 计算出的掩码
            else:
                base_update_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)
            
            final_update_mask = base_update_mask

            # =========================================================================
            # ======================== 【新增】强制自回归进展逻辑 =====================
            # =========================================================================
            if force_ar_progression and is_mask_mask.any():
                # 1. 找到每个batch中第一个MASK的位置
                # torch.argmax 在遇到全False的行时会返回0，但我们的逻辑稍后会处理
                first_mask_indices = torch.argmax(is_mask_mask.int(), dim=1)

                # 2. 创建一个只在第一个MASK位置为True的掩码
                forced_ar_mask = torch.zeros_like(is_mask_mask, dtype=torch.bool)
                forced_ar_mask.scatter_(1, first_mask_indices.unsqueeze(1), True)
                
                # 3. 关键：确保这个强制掩码只在原来就有MASK的行生效
                # 这样可以防止在没有MASK的行中，错误地将位置0标记为更新
                forced_ar_mask = forced_ar_mask & is_mask_mask.any(dim=1, keepdim=True)
                
                # 4. 使用逻辑“或”合并掩码，确保第一个MASK位置一定被更新
                final_update_mask = final_update_mask | forced_ar_mask
            # =========================================================================

            # 使用最终的掩码更新序列
            new_tokens = torch.where(final_update_mask, candidate_tokens, current_gen_part)
            current_sequence[:, generation_start_idx:] = new_tokens
            
            # ... (debug 和提前终止逻辑保持不变) ...
            if not (current_sequence == mask_token_id).any():
                if debug: logger.info("🎉 所有MASK已被替换，提前终止扩散过程。")
                break

        if debug:
            logger.info("\n" + "=" * 80)
            logger.info("🎉 并行扩散生成完成!")
            # ... (final debug info)
        
        return current_sequence

    def _generate_blockwise(
        self,
        input_ids: torch.LongTensor,
        mask_token_id: Optional[int],
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        num_diffusion_steps: int,
        temperature: float,
        use_token_change_classifier: bool,
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        debug: bool,
        tokenizer,
        decode_top_k_positions: Optional[int],
        block_size: int,
        force_ar_progression: bool, # 【新增】接收新参数
        **kwargs,
    ) -> torch.LongTensor:
        """
        新的逐块(block-wise)解码逻辑。
        自左向右，一次解码一个大小为 block_size 的块。
        """
        batch_size, _ = input_ids.shape
        device = input_ids.device

        # 初始化将随时间增长的序列和注意力掩码
        current_sequence = input_ids
        if attention_mask is None:
            # 如果没有提供 attention_mask，则创建一个全为1的掩码
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        current_attention_mask = attention_mask
        
        num_generated_tokens = 0

        if debug:
            logger.info("=" * 80)
            logger.info(f"🚀 开始逐块扩散生成过程")
            logger.info(f"📊 参数: max_new_tokens={max_new_tokens}, num_diffusion_steps={num_diffusion_steps}, block_size={block_size}")
            logger.info(f"🎯 采样设置: do_sample={do_sample}, temperature={temperature}")
            # ... (more debug info)
        
        # --- 外部循环：按块进行生成 ---
        while num_generated_tokens < max_new_tokens:
            # 1. 计算当前块的实际大小，处理最后一个可能不足一个block_size的块
            current_block_size = min(block_size, max_new_tokens - num_generated_tokens)
            if current_block_size <= 0: break # 如果计算出的块大小为0，则结束
                
            # 当前块的生成起始位置，即已生成序列的长度
            prompt_len_for_block = current_sequence.shape[1]
            
            if debug:
                logger.info("\n" + "#" * 80)
                logger.info(f"🧩 解码新块 (生成 {num_generated_tokens+1} 到 {num_generated_tokens+current_block_size} 的 tokens)")
                logger.info(f"   - 当前序列总长: {prompt_len_for_block}")
                logger.info(f"   - 本次块大小: {current_block_size}")
                logger.info("#" * 80)

            # 2. 准备此块扩散所需的序列和掩码：在当前序列后附着 MASK
            block_sequence_to_refine = torch.cat([
                current_sequence,
                torch.full((batch_size, current_block_size), mask_token_id, dtype=input_ids.dtype, device=device)
            ], dim=1)
            block_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((batch_size, current_block_size), dtype=torch.long, device=device)
            ], dim=1)

            # --- 内部循环：对当前块进行扩散优化 ---
            for step in range(num_diffusion_steps):
                if debug:
                    logger.info(f"\n🔄 === 块内扩散步骤 {step + 1}/{num_diffusion_steps} ===")

                prev_sequence_for_debug = block_sequence_to_refine.clone()

                # 3. 核心模型调用
                outputs = self.forward(
                    input_ids=block_sequence_to_refine,
                    attention_mask=block_attention_mask,
                    return_dict=True,
                    causal=True
                )
                mlm_logits = outputs.logits
                
                # 关键：生成部分的起始索引是当前块的起始位置
                generation_start_idx = prompt_len_for_block  

                # 4. 为当前块切片 logits
                if mlm_logits.shape[1] > generation_start_idx:
                    mlm_logits_gen = mlm_logits[:, generation_start_idx - 1:-1, :]
                else:
                    mlm_logits_gen = mlm_logits[:, -1, :].unsqueeze(1)
                
                assert mlm_logits_gen.shape[1] == current_block_size, f"Logits slice shape mismatch. Expected {current_block_size}, got {mlm_logits_gen.shape[1]}"

                # 5. 为当前块生成候选 tokens (采样逻辑)
                if do_sample:
                    mlm_logits_gen = mlm_logits_gen / temperature
                    if top_k is not None:
                        # top-k 过滤
                        top_k_logits, top_k_indices = torch.topk(mlm_logits_gen, k=min(top_k, mlm_logits_gen.size(-1)))
                        mlm_logits_gen = torch.full_like(mlm_logits_gen, float('-inf'))
                        mlm_logits_gen.scatter_(-1, top_k_indices, top_k_logits)
                    if top_p is not None:
                        # top-p (nucleus) 过滤
                        sorted_logits, sorted_indices = torch.sort(mlm_logits_gen, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                        mlm_logits_gen = mlm_logits_gen.masked_fill(indices_to_remove, float('-inf'))
                    
                    probs = torch.softmax(mlm_logits_gen, dim=-1)
                    candidate_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, current_block_size)
                else:
                    candidate_tokens = torch.argmax(mlm_logits_gen, dim=-1)

                # 更新策略 (同样只应用于当前块)
                current_gen_part = block_sequence_to_refine[:, generation_start_idx:]

                if decode_top_k_positions is not None:
                    is_mask_mask = (current_gen_part == mask_token_id)
                    if not is_mask_mask.any():
                        if debug: logger.info("  [信息] 块内没有剩余的MASK token，跳过更新。")
                        break
                    probs = torch.softmax(mlm_logits_gen, dim=-1)
                    confidence_scores, _ = torch.max(probs, dim=-1)
                    masked_confidence_scores = torch.where(is_mask_mask, confidence_scores, -1.0)
                    k = min(decode_top_k_positions, current_block_size)
                    _, top_k_indices = torch.topk(masked_confidence_scores, k=k, dim=1)
                    final_update_mask = torch.zeros_like(confidence_scores, dtype=torch.bool, device=device)
                    final_update_mask.scatter_(1, top_k_indices, True)
                    final_update_mask = final_update_mask & is_mask_mask
                    new_tokens_for_block = torch.where(final_update_mask, candidate_tokens, current_gen_part)
                    block_sequence_to_refine[:, generation_start_idx:] = new_tokens_for_block
                elif use_token_change_classifier:
                    if step == 0:
                        final_update_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)
                    else:
                        corrector_logits = outputs.corrector_logits
                        if corrector_logits.shape[1] > generation_start_idx:
                            change_logits_gen = corrector_logits[:, generation_start_idx - 1:-1, :]
                        else:
                            change_logits_gen = corrector_logits[:, -1, :].unsqueeze(1)
                        assert change_logits_gen.shape[1] == current_block_size, "Corrector logits slice shape mismatch."
                        change_decisions = change_logits_gen.sigmoid().squeeze(-1) > 0.5
                        sentences_with_no_changes = ~torch.any(change_decisions, dim=1)
                        final_update_mask = change_decisions | sentences_with_no_changes.unsqueeze(1)
                    new_tokens_for_block = torch.where(final_update_mask, candidate_tokens, current_gen_part)
                    block_sequence_to_refine[:, generation_start_idx:] = new_tokens_for_block
                else: # 默认的简单更新策略
                    block_sequence_to_refine[:, generation_start_idx:] = candidate_tokens
                    final_update_mask = torch.ones_like(candidate_tokens, dtype=torch.bool, device=device)
                
                if debug:
                    mask_positions = block_sequence_to_refine[:, generation_start_idx:] == mask_token_id
                    self._debug_step_changes(step + 1, prev_sequence_for_debug, block_sequence_to_refine, candidate_tokens, final_update_mask, mask_positions, generation_start_idx, batch_size, tokenizer, mask_token_id)

                # 7. 检查当前块是否已全部生成，如果是，则提前结束内部循环
                if not (block_sequence_to_refine[:, generation_start_idx:] == mask_token_id).any():
                    if debug: logger.info("🎉 当前块内所有MASK已被替换，提前完成该块的扩散。")
                    break
                
                import pdb
                pdb.set_trace()
            # --- 块的扩散循环结束后 ---
            # 8. 更新主序列和注意力掩码，为下一个块做准备
            current_sequence = block_sequence_to_refine
            current_attention_mask = block_attention_mask
            num_generated_tokens += current_block_size
        
        if debug:
            logger.info("\n" + "=" * 80)
            logger.info("🎉 所有块解码完成!")
            for batch_idx in range(batch_size):
                final_text = self._tokens_to_text(current_sequence[batch_idx], tokenizer, mask_token_id)
                logger.info(f"📝 Batch {batch_idx} 最终序列: {final_text}")
            logger.info("=" * 80)
        
        return current_sequence

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        mask_token_id: Optional[int],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        num_diffusion_steps: int = 10,
        temperature: float = 1.0,
        use_token_change_classifier = True,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        debug: bool = False,
        tokenizer = None,
        block_size: Optional[int] = None,
        decode_top_k_positions = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Args:
            input_ids: 输入的token ids，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
            max_new_tokens: 要生成的新token数量 (L)
            num_diffusion_steps: 扩散迭代次数 (T)
            temperature: MLM采样的温度参数
            do_sample: 是否使用采样，False则使用贪心解码
            top_k: top-k采样参数
            top_p: top-p采样参数
            mask_token_id: mask token的id，如果为None则尝试自动获取
            debug: 是否启用调试模式，输出每步迭代的详细信息
            tokenizer: 用于将token id转换为文本的tokenizer（可选）
            block_size: (可选) 按块生成的大小。如果设置，则会自左向右逐块生成。
                        如果不设置或大于等于max_new_tokens，则使用并行模式一次性生成。

        Returns:
            生成的完整序列，形状为 (batch_size, original_seq_len + max_new_tokens)
        """

        # 解码模式应当被分成以下几种
        # 1. 因果块自回归，模型只有在去噪完一个块内所有的token时才被允许进行下一个块的推断（通过附着新的块预算）
        # 2. 全局自回归推断
        # 
        if block_size is None or block_size >= max_new_tokens:
            return self._generate_parallel(
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_diffusion_steps=num_diffusion_steps,
                temperature=temperature,
                use_token_change_classifier=use_token_change_classifier,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                debug=debug,
                tokenizer=tokenizer,
                decode_top_k_positions=decode_top_k_positions,
                **kwargs,
            )
        else:
            return self._generate_blockwise(
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_diffusion_steps=num_diffusion_steps,
                temperature=temperature,
                use_token_change_classifier=use_token_change_classifier,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                debug=debug,
                tokenizer=tokenizer,
                decode_top_k_positions=decode_top_k_positions,
                block_size=block_size,
                **kwargs,
            )

    
    def _debug_step_changes(
        self,
        step: int,
        prev_sequence: torch.Tensor,
        current_sequence: torch.Tensor,
        candidate_tokens: torch.Tensor,
        change_decisions: torch.Tensor,
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
        actual_changes = prev_gen_tokens != curr_gen_tokens  # (batch_size, generation_len)

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
                logger.info(change_decisions)
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