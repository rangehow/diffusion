python generate_bd3lm.py \
    --model_dir /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/bd3lm_110m_lm1b_ema_fixed_1217_2node/checkpoint-1000000 \
    --gen_length 128 \
    --block_length 16 \
    --denoising_steps 8 \
    --use_kv_cache \
    --trust_remote_code