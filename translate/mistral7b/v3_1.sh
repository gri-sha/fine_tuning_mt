#!/bin/bash

python3 translate/translation.py \
    --checkpoint-path "sft_checkpoints/mistral7b/v3_v4/checkpoint-591_v3" \
    --translations-path "results/translations/mistral7b/v3_1.csv" \
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
