"""
Diffusion Generation Debugger (Optimized)
Visualizes the token generation process step by step
GPU-CPU transfers are deferred until after generation completes for better performance.
"""

import torch
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel


@dataclass
class StepRecord:
    """Record of a single generation step."""
    block_num: int
    step: int
    ar_pointer: int
    block_state: List[int]
    changed_positions: List[int]
    confidences: List[float]
    candidates: List[int]


@dataclass 
class BlockRecord:
    """Record of a completed block."""
    block_num: int
    final_tokens: List[int]
    generation_order: List[int]
    steps: List[StepRecord] = field(default_factory=list)


@dataclass
class TensorStepRecord:
    """Lightweight record that keeps tensors on GPU during generation."""
    block_num: int
    step: int
    ar_pointer: int
    block_state: torch.Tensor  # Keep on GPU
    old_block_state: torch.Tensor  # Keep on GPU
    confidences: torch.Tensor  # Keep on GPU
    candidates: torch.Tensor  # Keep on GPU


class DiffusionDebugger:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        add_bos: bool = False,
    ):
        self.device = device
        self.add_bos = add_bos

        print("=" * 80)
        print("Loading model...")
        print("=" * 80)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(device).eval()

        self.mask_token_id = self.model.config.mask_token_id
        self.mask_token = self.tokenizer.decode([self.mask_token_id])

        print(f"✓ Model loaded")
        print(f"✓ Mask token: '{self.mask_token}' (id={self.mask_token_id})")
        print()

    def _colorize(self, text: str, color: str) -> str:
        """Add ANSI color codes to text."""
        colors = {
            'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
            'blue': '\033[94m', 'magenta': '\033[95m', 'cyan': '\033[96m',
            'white': '\033[97m', 'bold': '\033[1m', 'dim': '\033[2m',
            'reset': '\033[0m',
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def _convert_tensor_records_to_block_records(
        self,
        tensor_records: List[List[TensorStepRecord]],
        block_final_tokens: List[torch.Tensor],
        block_gen_orders: List[List[int]],
    ) -> List[BlockRecord]:
        """
        Convert GPU tensor records to CPU BlockRecords after generation completes.
        This is where all the GPU-CPU transfers happen, but only once at the end.
        """
        all_records = []
        
        for block_idx, (step_records, final_tokens, gen_order) in enumerate(
            zip(tensor_records, block_final_tokens, block_gen_orders)
        ):
            block_record = BlockRecord(
                block_num=block_idx + 1,
                final_tokens=final_tokens.tolist(),
                generation_order=gen_order,
            )
            
            for tensor_rec in step_records:
                block_state = tensor_rec.block_state.tolist()
                old_state = tensor_rec.old_block_state.tolist()
                conf_list = tensor_rec.confidences.tolist()
                cand_list = tensor_rec.candidates.tolist()
                
                # Compute changed positions on CPU
                changed_positions = [
                    pos for pos in range(len(block_state))
                    if pos >= tensor_rec.ar_pointer and block_state[pos] != old_state[pos]
                ]
                
                block_record.steps.append(StepRecord(
                    block_num=tensor_rec.block_num,
                    step=tensor_rec.step,
                    ar_pointer=tensor_rec.ar_pointer,
                    block_state=block_state,
                    changed_positions=changed_positions,
                    confidences=conf_list,
                    candidates=cand_list,
                ))
            
            all_records.append(block_record)
        
        return all_records

    def _visualize_block_record(self, record: BlockRecord):
        """Visualize a block record after generation."""
        print(f"\n{'─' * 100}")
        print(self._colorize(f"📦 BLOCK {record.block_num} (size={len(record.final_tokens)})", 'cyan'))
        print('─' * 100)

        for step_rec in record.steps:
            print(f"\n  {self._colorize(f'Step {step_rec.step}', 'bold')} (AR pointer @ {step_rec.ar_pointer}):")

            # Show block state with changes highlighted
            tokens_viz = []
            for pos, tok_id in enumerate(step_rec.block_state):
                is_mask = (tok_id == self.mask_token_id)
                is_changed = pos in step_rec.changed_positions
                conf = step_rec.confidences[pos] if pos < len(step_rec.confidences) else 0.0

                if is_mask:
                    tok_str = self._colorize("[M]", 'dim')
                elif is_changed:
                    tok_text = self.tokenizer.decode([tok_id]).replace('\n', '↵').replace(' ', '␣')
                    tok_str = self._colorize(f"「{tok_text}」", 'green') + f"({conf:.2f})"
                else:
                    tok_text = self.tokenizer.decode([tok_id]).replace('\n', '↵').replace(' ', '␣')
                    tok_str = tok_text

                if pos == step_rec.ar_pointer:
                    tok_str = self._colorize("▶", 'yellow') + tok_str

                tokens_viz.append(tok_str)

            print(f"    State: {' '.join(tokens_viz)}")
            print(f"    Changed: {step_rec.changed_positions if step_rec.changed_positions else 'none'}")

        # Block summary
        block_text = self.tokenizer.decode(record.final_tokens)
        print(f"\n  {self._colorize('✅ Block committed:', 'cyan')} \"{block_text}\"")
        print(f"  Generation order: {record.generation_order}")

    def _visualize_summary(self, all_records: List[BlockRecord], gen_tokens: List[int],
                           generation_order: List[int], elapsed: float, prefix_len: int,
                           input_ids: torch.Tensor):
        """Visualize final summary."""
        max_new_tokens = len(gen_tokens)
        final_output = input_ids[0].tolist() + gen_tokens
        generated_text = self.tokenizer.decode(gen_tokens)
        full_text = self.tokenizer.decode(final_output)

        print("\n" + "=" * 100)
        print(self._colorize("📊 GENERATION SUMMARY", 'bold'))
        print("=" * 100)

        print(f"\n🎯 Generated text: \"{generated_text}\"")
        print(f"\n📜 Full output: \"{full_text}\"")

        print(f"\n⏱️ Performance:")
        print(f"   Total time: {elapsed:.3f}s")
        print(f"   Tokens generated: {max_new_tokens}")
        print(f"   Tokens/second: {max_new_tokens / elapsed:.1f}")

        # Generation order visualization
        print(f"\n🔢 Token Generation Order Map:")
        print("   (Number indicates the step when token was first assigned)")
        print()

        for start in range(0, max_new_tokens, 8):
            end = min(start + 8, max_new_tokens)
            tokens_line = []
            orders_line = []
            for pos in range(start, end):
                tok_text = self.tokenizer.decode([gen_tokens[pos]]).replace('\n', '↵')
                if len(tok_text) > 8:
                    tok_text = tok_text[:6] + ".."
                tokens_line.append(f"{tok_text:^10}")
                order = generation_order[pos]
                orders_line.append(f"{'['+str(order)+']':^10}" if order >= 0 else f"{'[?]':^10}")

            print(f"   Pos {start:3d}-{end-1:3d}: {' '.join(tokens_line)}")
            print(f"   {'':8}  {' '.join(orders_line)}")
            print()

        # Step-by-step timeline
        print(f"\n📽️ Step-by-step token reveal timeline:")
        step_to_positions = {}
        for pos, order in enumerate(generation_order):
            if order >= 0:
                step_to_positions.setdefault(order, []).append(pos)

        for step in sorted(step_to_positions.keys()):
            positions = step_to_positions[step]
            tokens_at_step = [self.tokenizer.decode([gen_tokens[p]]).replace('\n', '↵') for p in positions]
            print(f"   Step {step:3d}: positions {positions} -> {tokens_at_step}")

    @torch.inference_mode()
    def debug_generate_block_kvcache(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        steps: int = 8,
        temperature: float = 1.0,
        block_size: int = 8,
    ) -> str:
        """
        Debug generation with detailed step-by-step visualization.
        Optimized: collects tensor references during generation, converts to CPU after completion.
        """
        print("\n" + "=" * 100)
        print(self._colorize("🔍 DIFFUSION BLOCK GENERATION DEBUG (OPTIMIZED)", 'bold'))
        print("=" * 100)
        print(f"Prompt: \"{prompt}\"")
        print(f"Config: max_new_tokens={max_new_tokens}, steps={steps}, block_size={block_size}, temp={temperature}")
        print("=" * 100)

        # Tokenize
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
            add_special_tokens=self.add_bos
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch_size, prefix_len = input_ids.shape
        device, dtype = input_ids.device, self.model.lm_head.weight.dtype

        print(f"\n📝 Prefix tokens ({prefix_len}):")
        prefix_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        for i, (tok, tok_id) in enumerate(zip(prefix_tokens, input_ids[0].tolist())):
            print(f"  [{i:3d}] {tok_id:6d} -> '{tok}'")

        # Initialize KV cache
        cache = self.model._init_kv_cache(batch_size, prefix_len + max_new_tokens, device, dtype)

        # Cache prefix
        print(f"\n⚡ Caching prefix...")
        self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache.get_all_layer_caches(),
            cache_seqlens=cache.get_cache_seq_lens(),
            use_cache=True, causal=True, return_dict=True
        )
        cache.advance_seq_len(prefix_len)

        # Generation state
        committed = input_ids.clone()
        committed_len = prefix_len
        all_generated = []
        num_generated = 0
        global_step = 0
        block_num = 0

        # Lightweight tensor records (stays on GPU during generation)
        all_tensor_records: List[List[TensorStepRecord]] = []
        all_block_final_tokens: List[torch.Tensor] = []
        all_block_gen_orders: List[List[int]] = []

        print(f"\n🚀 Starting generation...")
        start_time = time.perf_counter()

        while num_generated < max_new_tokens:
            block_num += 1
            curr_block_size = min(block_size, max_new_tokens - num_generated)

            # Initialize block
            block = torch.full(
                (batch_size, curr_block_size),
                self.mask_token_id,
                dtype=input_ids.dtype,
                device=device
            )

            block_gen_order = [-1] * curr_block_size
            block_tensor_records: List[TensorStepRecord] = []

            # Block iterations
            for step in range(steps):
                ar_pointer = step
                if ar_pointer >= curr_block_size:
                    break

                # Forward pass
                cache.set_seq_len(committed_len - 1)
                block_input = torch.cat([committed[:, -1:], block], dim=1)

                outputs = self.model.forward(
                    input_ids=block_input,
                    past_key_values=cache.get_all_layer_caches(),
                    cache_seqlens=cache.get_cache_seq_lens(),
                    use_cache=True, causal=True, return_dict=True
                )

                relevant_logits = outputs.logits[:, :curr_block_size, :]
                probs = torch.softmax(relevant_logits.float(), dim=-1)
                confidences, _ = probs.max(dim=-1)

                # Sample
                if temperature > 0:
                    gumbel_logits = self.model._add_gumbel_noise(relevant_logits, temperature)
                    candidate_tokens = torch.argmax(gumbel_logits, dim=-1).to(input_ids.dtype)
                else:
                    candidate_tokens = torch.argmax(relevant_logits, dim=-1).to(input_ids.dtype)

                # Store old block state BEFORE update (clone to preserve)
                old_block = block.clone()

                # Update
                update_mask = torch.zeros((batch_size, curr_block_size), dtype=torch.bool, device=device)
                update_mask[:, ar_pointer:] = True
                block = torch.where(update_mask, candidate_tokens, old_block)

                # Store tensor record (NO .tolist() calls - stays on GPU)
                block_tensor_records.append(TensorStepRecord(
                    block_num=block_num,
                    step=step,
                    ar_pointer=ar_pointer,
                    block_state=block[0].clone(),  # Clone to preserve state
                    old_block_state=old_block[0].clone(),
                    confidences=confidences[0].clone(),
                    candidates=candidate_tokens[0].clone(),
                ))

                # Update generation order (only need comparison on GPU, result is small)
                # Use GPU comparison, then minimal transfer
                changed_mask = (block[0] != old_block[0]) & (torch.arange(curr_block_size, device=device) >= ar_pointer)
                changed_positions = changed_mask.nonzero(as_tuple=True)[0].tolist()
                
                for pos in changed_positions:
                    if block_gen_order[pos] == -1:
                        block_gen_order[pos] = global_step

                global_step += 1

                # Early exit check (stays on GPU)
                if step > 0 and not changed_mask.any():
                    break

            # Commit block
            cache.set_seq_len(committed_len - 1)
            self.model.forward(
                input_ids=torch.cat([committed[:, -1:], block], dim=1),
                past_key_values=cache.get_all_layer_caches(),
                cache_seqlens=cache.get_cache_seq_lens(),
                use_cache=True, causal=True, return_dict=True
            )
            cache.set_seq_len(committed_len + curr_block_size)

            committed_len += curr_block_size
            committed = torch.cat([committed, block], dim=1)
            all_generated.append(block)

            # Store block data (tensor stays on GPU)
            all_tensor_records.append(block_tensor_records)
            all_block_final_tokens.append(block[0].clone())
            all_block_gen_orders.append(block_gen_order)

            num_generated += curr_block_size

        # Synchronize before timing
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # === NOW do all GPU-CPU transfers (after generation is complete) ===
        print(f"\n✅ Generation complete in {elapsed:.3f}s! Converting data for visualization...")
        
        # Convert tensor records to CPU BlockRecords
        all_records = self._convert_tensor_records_to_block_records(
            all_tensor_records, all_block_final_tokens, all_block_gen_orders
        )

        # Collect all generation orders
        generation_order = []
        for gen_order in all_block_gen_orders:
            generation_order.extend(gen_order)

        # Visualize
        for record in all_records:
            self._visualize_block_record(record)

        gen_tokens = []
        for record in all_records:
            gen_tokens.extend(record.final_tokens)

        self._visualize_summary(all_records, gen_tokens, generation_order, 
                                elapsed, prefix_len, input_ids)

        return self.tokenizer.decode(gen_tokens)

    @torch.inference_mode()
    def debug_generate_prefix_cache(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        num_diffusion_steps: int = 8,
        temperature: float = 1.0,
        top_p: float = 0.95,
        decode_top_k_positions: Optional[int] = None,
    ) -> str:
        """Debug the prefix cache generation method (optimized)."""
        print("\n" + "=" * 100)
        print(self._colorize("🔍 DIFFUSION PREFIX-CACHE GENERATION DEBUG (OPTIMIZED)", 'bold'))
        print("=" * 100)
        print(f"Prompt: \"{prompt}\"")
        print(f"Config: max_new_tokens={max_new_tokens}, steps={num_diffusion_steps}, temp={temperature}, top_p={top_p}")
        if decode_top_k_positions:
            print(f"        decode_top_k_positions={decode_top_k_positions}")
        print("=" * 100)

        # Tokenize
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
            add_special_tokens=self.add_bos
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch_size, prefix_len = input_ids.shape
        device = input_ids.device

        print(f"\n📝 Prefix length: {prefix_len} tokens")

        # Initialize with masks
        gen_tokens = torch.full((batch_size, max_new_tokens), self.mask_token_id, 
                                dtype=input_ids.dtype, device=device)

        generation_order = [-1] * max_new_tokens
        global_step = 0

        # Collect tensor records (stays on GPU during generation)
        tensor_step_records: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

        print(f"\n🚀 Starting generation...")
        start_time = time.perf_counter()

        for step in range(num_diffusion_steps):
            # Forward pass
            full_seq = torch.cat([input_ids, gen_tokens], dim=1)
            full_mask = torch.cat([
                attention_mask, 
                torch.ones((batch_size, max_new_tokens), dtype=attention_mask.dtype, device=device)
            ], dim=1) if attention_mask is not None else None

            outputs = self.model.forward(
                input_ids=full_seq, attention_mask=full_mask,
                use_cache=False, causal=True, return_dict=True
            )

            gen_logits = outputs.logits[:, prefix_len - 1:-1, :]
            probs = torch.softmax(gen_logits.float(), dim=-1)
            confidences, _ = probs.max(dim=-1)

            # Sample
            candidates = sample_from_logits(gen_logits, temperature, None, top_p, do_sample=True)

            is_mask = (gen_tokens == self.mask_token_id)

            # Update mask
            if decode_top_k_positions is not None and is_mask.any():
                masked_conf = torch.where(is_mask, confidences, torch.tensor(-1.0, device=device))
                k = min(decode_top_k_positions, is_mask.sum().item())
                _, top_idx = masked_conf.topk(k, dim=1)
                update_mask = torch.zeros_like(is_mask).scatter_(1, top_idx, True) & is_mask
            else:
                update_mask = torch.ones_like(is_mask)

            old_gen = gen_tokens.clone()
            gen_tokens = torch.where(update_mask, candidates, gen_tokens)

            # Store tensor record (NO .tolist() - stays on GPU)
            mask_count = is_mask.sum().item()  # Small scalar, OK to transfer
            tensor_step_records.append((
                old_gen[0].clone(),
                gen_tokens[0].clone(),
                confidences[0].clone(),
                mask_count,
            ))

            # Update generation order using GPU comparison
            changed_mask = (gen_tokens[0] != old_gen[0])
            changed_positions = changed_mask.nonzero(as_tuple=True)[0].tolist()

            for pos in changed_positions:
                if generation_order[pos] == -1:
                    generation_order[pos] = global_step

            global_step += 1

            # Early exit check (on GPU)
            if not (gen_tokens == self.mask_token_id).any():
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # === NOW do all GPU-CPU transfers (after generation complete) ===
        print(f"\n✅ Generation complete in {elapsed:.3f}s! Converting data for visualization...")

        # Convert tensor records to CPU
        step_records = []
        for step_idx, (old_state_t, new_state_t, conf_t, mask_count) in enumerate(tensor_step_records):
            old_state = old_state_t.tolist()
            new_state = new_state_t.tolist()
            conf_list = conf_t.tolist()
            
            changed = [pos for pos in range(max_new_tokens) if old_state[pos] != new_state[pos]]
            
            step_records.append({
                'step': step_idx,
                'state': new_state,
                'changed': changed,
                'confidences': conf_list,
                'mask_count': mask_count,
            })

        # Visualization
        for rec in step_records:
            print(f"\n{'─' * 100}")
            print(self._colorize(f"📍 STEP {rec['step'] + 1}", 'cyan'))
            print(f"  Remaining masks: {rec['mask_count']}/{max_new_tokens}")
            print(f"  Updated positions: {rec['changed'][:20]}{'...' if len(rec['changed']) > 20 else ''}")

            # Show first 40 tokens
            show_len = min(40, max_new_tokens)
            state_viz = []
            for pos in range(show_len):
                tok_id = rec['state'][pos]
                if tok_id == self.mask_token_id:
                    state_viz.append(self._colorize("[M]", 'dim'))
                else:
                    tok_text = self.tokenizer.decode([tok_id]).replace('\n', '↵').replace(' ', '␣')
                    if len(tok_text) > 4:
                        tok_text = tok_text[:3] + "."
                    if pos in rec['changed']:
                        state_viz.append(self._colorize(f"「{tok_text}」", 'green'))
                    else:
                        state_viz.append(tok_text)

            print(f"  State (0-{show_len-1}): {' '.join(state_viz)}")

        # Final summary
        gen_tokens_list = gen_tokens[0].tolist()
        generated_text = self.tokenizer.decode(gen_tokens_list)
        full_text = self.tokenizer.decode(input_ids[0].tolist() + gen_tokens_list)

        print("\n" + "=" * 100)
        print(self._colorize("📊 GENERATION SUMMARY", 'bold'))
        print("=" * 100)

        print(f"\n🎯 Generated text: \"{generated_text}\"")
        print(f"\n📜 Full output: \"{full_text}\"")

        print(f"\n⏱️ Performance:")
        print(f"   Total time: {elapsed:.3f}s")
        print(f"   Tokens generated: {max_new_tokens}")
        print(f"   Tokens/second: {max_new_tokens / elapsed:.1f}")

        # Generation order
        print(f"\n🔢 Token Generation Order Map:")
        for start in range(0, max_new_tokens, 8):
            end = min(start + 8, max_new_tokens)
            tokens_line = []
            orders_line = []
            for pos in range(start, end):
                tok_text = self.tokenizer.decode([gen_tokens_list[pos]]).replace('\n', '↵')
                if len(tok_text) > 8:
                    tok_text = tok_text[:6] + ".."
                tokens_line.append(f"{tok_text:^10}")
                order = generation_order[pos]
                orders_line.append(f"{'['+str(order)+']':^10}" if order >= 0 else f"{'[?]':^10}")

            print(f"   Pos {start:3d}-{end-1:3d}: {' '.join(tokens_line)}")
            print(f"   {'':8}  {' '.join(orders_line)}")
            print()

        return generated_text


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, do_sample=True):
    """Sample from logits with temperature and top-p."""
    if not do_sample:
        return logits.argmax(dim=-1)

    shape, vocab = logits.shape[:-1], logits.shape[-1]
    flat = logits.reshape(-1, vocab) / (temperature if temperature != 1.0 else 1.0)

    if top_k and top_k > 0:
        threshold = flat.topk(min(top_k, vocab), dim=-1).values[:, -1:]
        flat = flat.masked_fill(flat < threshold, float('-inf'))

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = flat.sort(dim=-1, descending=True)
        cumsum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumsum - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits[remove] = float('-inf')
        flat = sorted_logits.gather(-1, sorted_idx.argsort(-1))

    return torch.multinomial(flat.softmax(dim=-1), 1).squeeze(-1).reshape(shape)


def main():
    MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp/checkpoint-77335"
    DEVICE = "cuda"
    ADD_BOS = False

    TEST_PROMPTS = [
        "The meaning of life is",
    ]

    debugger = DiffusionDebugger(
        model_path=MODEL_PATH,
        device=DEVICE,
        add_bos=ADD_BOS,
    )

    for prompt in TEST_PROMPTS:
        print("\n" + "🌟" * 50)

        # Block KV cache method
        debugger.debug_generate_block_kvcache(
            prompt=prompt,
            max_new_tokens=32,
            steps=8,
            temperature=1.0,
            block_size=8,
        )

        print("\n" + "═" * 100)

        # Prefix cache method  
        debugger.debug_generate_prefix_cache(
            prompt=prompt,
            max_new_tokens=32,
            num_diffusion_steps=8,
            temperature=1.0,
            top_p=0.95,
            decode_top_k_positions=8,
        )


if __name__ == "__main__":
    main()