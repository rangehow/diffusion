rsync -av --info=progress2 \
--exclude 'optimizer.pt' \
--exclude 'scheduler.pt' \
--exclude 'rng_state_*.pth' \
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_1B_100b_2node/checkpoint-309339 \
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/old_niu_1b_100b_2node/


rsync -av --info=progress2 \
--exclude 'optimizer.pt' \
--exclude 'scheduler.pt' \
--exclude 'rng_state_*.pth' \
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/checkpoint-309339 \
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/


rsync -av --info=progress2 \
--exclude 'optimizer.pt' \
--exclude 'scheduler.pt' \
--exclude 'rng_state_*.pth' \
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_loose_2node/checkpoint-309339 \
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_loose_2node/

rsync -av --info=progress2 \
--exclude 'optimizer.pt' \
--exclude 'scheduler.pt' \
--exclude 'rng_state_*.pth' \
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_1B_tight_2node/checkpoint-309339 \
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_tight_2node/

# rsync -av --info=progress2 \
# --exclude 'optimizer.pt' \
# --exclude 'scheduler.pt' \
# --exclude 'ema_state.pt' \
# --exclude 'rng_state_*.pth' \
# /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/mdlm_1B_fineweb_edu_100b_potential \
# /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_1B_fineweb_edu_100b_potential/

# rsync -av --info=progress2 \
# --exclude 'optimizer.pt' \
# --exclude 'scheduler.pt' \
# --exclude 'ema_state.pt' \
# --exclude 'rng_state_*.pth' \
# /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_1B_fineweb_edu_100b_potential \
# /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_fineweb_edu_100b_potential/