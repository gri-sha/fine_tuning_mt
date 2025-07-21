#!/bin/bash

OUTPUT_DIR="sft_checkpoints/mistral7b/v1"
mkdir -p "$OUTPUT_DIR"

python3 finetuning.py \
    --model-name "mistralai/Mistral-7B-v0.1" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --learning-rate 2e-3 \
    --batch-size 32 \
    --max-seq-length 512 \
    --logging-steps 20 \
    --completion-only-loss "false" \
    --warmup-steps 0 \
    --eval-strategy "epoch" \
    --eval-steps "" \
    --seed 1212 \
    --test-split 0.1 \
    --valid-split 0.05 \
    --shots "0 1" \
    --fuzzy "f t" \
    --bos_token "true" \
    --eos-token "false" \
    --pad-side "right" \
    --packing "true" \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-rank 64 \
    --lora-bias "none" \
    --lora-task "CAUSAL_LM"
