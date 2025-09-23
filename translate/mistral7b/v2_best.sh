#!/bin/bash

python3 translate/translation.py \
    --checkpoint-path "sft_checkpoints/mistral7b/v2/best_loss_step_512" \
    --translations-path "results/translations/mistral7b/v2_best.csv" \
    --shots 0 1\
    --fuzzy _ t\
    --max-new-tokens 128 \
    --batch-size 16 \
    --use-batches true \
    --add-bos-token true \
    --add-eos-token false \
    --use-fast-tokenizer true \
    --pad-side left \
    --pad-token "</s>" \
    --do-inference-check true \
    --do-translation true \
