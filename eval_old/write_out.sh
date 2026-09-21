cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/logprob_eval

# python write_out.py \
#   --model_name_or_path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2-0.5B/main \
#   --tasks arc_challenge \
#   --num_fewshot 0 \
#   --limit 0 \
#   --output_dir results


python lm_eval_write_out.py \
    --tasks winogrande \
    --num_fewshot 1 \
    --output_base_path ./prompts