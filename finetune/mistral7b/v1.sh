#!/bin/bash

# Mistral-7B-v0.3 is a decoder-only model: packing is not used due to issues of compatibility of flash attention 2 with versions torch and cuda
# The default pad token is the same as the eos token ('</s>'), we keep it that way
# With this tokenizer SFTTrainer automatically adds eos token, so we set 'add_eos_token' to false
# Bos token is applicable, we set 'add_bos_token' to true
# For decoder only model padding side during training is right, but during inference it is left

# (bos token <s> , eos token <\s>, pad token <\s>)


OUTPUT_DIR="sft_checkpoints/mistral7b/v1"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/output.log") 2>&1


python3 finetune/finetuning.py \
    --model-name mistralai/Mistral-7B-v0.3 \
    --output-dir "$OUTPUT_DIR" \
    --shots 0 \
    --fuzzy _ \
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
    --learning-rate 2e-4 \
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