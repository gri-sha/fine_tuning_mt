#!/bin/bash

OUTPUT_DIR="sft_checkpoints/mt5/v2"
mkdir -p "$OUTPUT_DIR"
exec >"$OUTPUT_DIR/output.log" 2>&1

python3 finetune/finetuning.py \
    --model-name "google/mt5-xl" \
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
    --lora-task "SEQ_2_SEQ_LM" \
    --epochs 2 \
    --learning-rate 2e-4 \
    --batch-size 32 \
    --packing "false" \
    --bf16-for-compute "true" \
    --max-seq-length 512 \
    --logging-steps 32 \
    --completion-only-loss "true" \
    --warmup-steps 0 \
    --eval-strategy "steps" \
    --eval-steps 96 \
    --save-strategy "epoch"

# max_tokens for translation of the test dataset: 30