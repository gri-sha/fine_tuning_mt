#!/bin/bash

OUTPUT_DIR="sft_checkpoints/mt5/v2"
mkdir -p "$OUTPUT_DIR"

python3 finetune/finetuning.py \
    --model-name "google/mt5-xl" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 2 \
    --learning-rate 2e-3 \
    --batch-size 64 \
    --max-seq-length 512 \
    --logging-steps 32 \
    --completion-only-loss "true" \
    --warmup-steps 0 \
    --eval-strategy "steps" \
    --eval-steps 96 \
    --shots "0 1" \
    --fuzzy "f t" \
    --bos_token "true" \
    --eos-token "true" \
    --pad-side "right" \
    --packing "false" \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-rank 64 \
    --lora-bias "none" \
    --lora-task "CAUSAL_LM"