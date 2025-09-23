#!/bin/bash

python3 translate/translation.py \
    --checkpoint-path "sft_checkpoints/mt5/v1/best_loss_step_1536" \
    --translations-path "results/translations/mt5/v1_best.csv" \
    --shots 0 1\
    --fuzzy _ 1\
    --max-new-tokens 128 \
    --batch-size 16 \
    --use-batches true \
    --add-eos-token true \
    --use-fast-tokenizer false \
    --pad-side right \
    --do-tokenizer-check true \
    --do-translation false \