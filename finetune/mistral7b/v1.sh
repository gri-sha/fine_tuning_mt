#!/bin/bash

OUTPUT_DIR="sft_checkpoints/mistral7b/v1"
mkdir -p "$OUTPUT_DIR"

python3 finetune/finetuning.py \
    --model-name "mistralai/Mistral-7B-v0.1" \
    --output-dir "$OUTPUT_DIR" \
    --shots "0 1" \
    --fuzzy "f t" \
    --bos_token "true" \
    --eos-token "false" \
    --pad-side "right" \
    --quantization "4bit" \
    --double_quant "true" \
    --quant-type "nf4" \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-rank 64 \
    --lora-bias "none" \
    --lora-task "CAUSAL_LM" \
    --epochs 1 \
    --learning-rate 2e-3 \
    --batch-size 32 \
    --packing "true" \
    --bf16-for-compute "true" \
    --max-seq-length 512 \
    --logging-steps 20 \
    --completion-only-loss "false" \
    --warmup-steps 0 \
    --eval-strategy "epoch" \


# max_tokens for translation of the test dataset: 30