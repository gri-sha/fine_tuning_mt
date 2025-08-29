#!/bin/bash

OUTPUT_DIR="sft_checkpoints/mt5/v1"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/output.log") 2>&1


python3 finetune/finetuning.py \
    --model-name google/umt5-xl \
    --output-dir "$OUTPUT_DIR" \
    --shots 0 \
    --fuzzy _ \
    --add-eos-token false \
    --use-fast-tokenizer false \
    --quantization 4bit \
    --double-quant true \
    --quant-type nf4 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-rank 64 \
    --lora-bias none \
    --lora-task SEQ_2_SEQ_LM \
    --lora-targets q v \
    --epochs 1 \
    --learning-rate 2e-4 \
    --batch-size 16 \
    --packing true \
    --bf16-for-compute true \
    --max-seq-length 512 \
    --logging-steps 32 \
    --completion-only-loss true \
    --warmup-steps 0 \
    --eval-strategy steps \
    --eval-steps 96 \
    --save-strategy epoch \
    --do-tokenizer-check true \
    --do-training false \
    --do-evaluation true 