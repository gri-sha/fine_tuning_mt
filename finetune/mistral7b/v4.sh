#!/bin/bash

# misleading configuration for the model 

OUTPUT_DIR="sft_checkpoints/mistral7b/v4"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/output.log") 2>&1

python3 finetune/finetuning.py \
    --model-name mistralai/Mistral-7B-v0.3 \
    --output-dir "$OUTPUT_DIR" \
    --shots 0 1 2\
    --fuzzy _ t t\
    --val-shots 0 1 \
    --val-fuzzy _ t \
    --add-bos-token true \
    --add-eos-token false \
    --use-fast-tokenizer true \
    --pad-side right \
    --quantization 4bit \
    --double-quant true \
    --quant-type nf4 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-rank 64 \
    --lora-bias none \
    --lora-task CAUSAL_LM \
    --epochs 2 \
    --learning-rate 2e-3 \
    --batch-size 32 \
    --gradient-accumulation-steps 1 \
    --packing false \
    --bf16-for-compute true \
    --max-seq-length 512 \
    --logging-steps 24 \
    --completion-only-loss true \
    --warmup-steps 0 \
    --eval-strategy steps \
    --eval-steps 64 \
    --save-strategy epoch \
    --do-tokenizer-check true \
    --do-training true \
    --do-evaluation true 