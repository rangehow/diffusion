#!/bin/bash

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/ppl/generation_outputs"

# Create merged folder
MERGED_DIR="${BASE_DIR}/hellaswag_origin_merged"
mkdir -p "$MERGED_DIR"

# Source folders to merge
FOLDERS=(
    "hellaswag_quality_cosine_b16_s8_20260128_151413"
    "hellaswag_quality_cosine_b16_s16_20260128_151533"
    "hellaswag_quality_cosine_b32_s8_20260128_151428"
    "hellaswag_quality_cosine_b64_s4_20260128_151600"
    "hellaswag_llama_20260128_151600"
)

# Copy contents from each folder
for folder in "${FOLDERS[@]}"; do
    src="${BASE_DIR}/${folder}"
    if [ -d "$src" ]; then
        echo "Copying from: $folder"
        cp -r "$src"/* "$MERGED_DIR"/
    else
        echo "Warning: $folder not found"
    fi
done

echo "Merge complete: $MERGED_DIR"