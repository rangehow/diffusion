# With Qwen3-0.6B
python calculate_ppl.py \
    --gen_dir generation_outputs/hellaswag_20241201_120000 \
    --eval_model /path/to/Qwen3-0.6B-Base \
    --eval_model_name Qwen3-0.6B

# With Llama-7B
python calculate_ppl.py \
    --gen_dir generation_outputs/hellaswag_20241201_120000 \
    --eval_model /path/to/Llama-7B \
    --eval_model_name Llama-7B

# With GPT-2
python calculate_ppl.py \
    --gen_dir generation_outputs/hellaswag_20241201_120000 \
    --eval_model gpt2-large \
    --eval_model_name GPT2-Large

# Only evaluate specific methods
python calculate_ppl.py \
    --gen_dir generation_outputs/hellaswag_20241201_120000 \
    --eval_model /path/to/model \
    --methods quality_cosine llama