#!/bin/bash

OUTPUT_DIR="sft_checkpoints/mistral7b/v5"
mkdir -p "$OUTPUT_DIR"

python3 finetune/finetuning.py \
    --model-name "mistralai/Mistral-7B-v0.3" \
    --output-dir "$OUTPUT_DIR" \
    --shots "0 1 2" \
    --fuzzy "f t t" \
    --bos_token "true" \
    --eos-token "true" \
    --pad-side "right" \
    --quantization "4bit" \
    --double_quant "true" \
    --quant-type "nf4" \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-rank 64 \
    --lora-bias "none" \
    --lora-task "CAUSAL_LM" \
    --epochs 3 \
    --learning-rate 2e-3 \
    --batch-size 64 \
    --packing "false" \
    --bf16-for-compute "true" \
    --max-seq-length 512 \
    --logging-steps 24 \
    --completion-only-loss "true" \
    --warmup-steps 0 \
    --eval-strategy "steps" \
    --eval-steps 64

# max_tokens for translation of the test dataset: 30