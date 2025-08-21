#!/bin/bash

OUTPUT_DIR="sft_checkpoints/qwen3-14b/v1"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/output.log") 2>&1


python3 finetune/finetuning.py \
    --model-name "Qwen/Qwen3-14B-Base" \
    --output-dir "$OUTPUT_DIR" \
    --shots "0 1 2" \
    --fuzzy "f t t" \
    --bos-token "true" \
    --eos-token "true" \
    --pad-side "right" \
    --quantization "4bit" \
    --double-quant "true" \
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
    --eval-steps 64 \
    --save-strategy "epoch"

# max_tokens for translation of the test dataset: 30