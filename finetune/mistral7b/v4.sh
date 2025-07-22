#!/bin/bash

OUTPUT_DIR="sft_checkpoints/mistral7b/v3"
mkdir -p "$OUTPUT_DIR"

python3 finetune/finetuning.py \
    --model-name "mistralai/Mistral-7B-v0.1" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 3 \
    --learning-rate 2e-3 \
    --batch-size 64 \
    --max-seq-length 512 \
    --logging-steps 24 \
    --completion-only-loss "true" \
    --warmup-steps 0 \
    --eval-strategy "steps" \
    --eval-steps 64 \
    --shots "0 1 2" \
    --fuzzy "f t t" \
    --bos_token "true" \
    --eos-token "false" \
    --pad-side "right" \
    --packing "true" \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-rank 64 \
    --lora-bias "none" \
    --lora-task "CAUSAL_LM"