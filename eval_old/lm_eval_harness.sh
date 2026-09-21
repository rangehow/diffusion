cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/eval
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2-0.5B/main,dtype="bfloat16" \
    --tasks mmlu   \
    --num_fewshot 5 \
    --batch_size 32