export HF_HOME=/opt/huggingface/grigory/
export HF_HUB_CACHE=/opt/huggingface/grigory/hub

python3 translate/translation.py \
    --model-name "mistralai/Mistral-7B-Instruct-v0.3" \
    --use-base-model true \
    --translations-path "results/translations/mistral7b/instruct.csv" \
    --shots 0 1\
    --fuzzy _ t\
    --max-new-tokens 128 \
    --batch-size 16 \
    --use-batches true \
    --add-bos-token true \
    --add-eos-token false \
    --use-fast-tokenizer false \
    --pad-side left \
    --pad-token '</s>' \
    --quantization 4bit \
    --double-quant true \
    --quant-type nf4 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --lora-rank 64 \
    --lora-bias none \
    --lora-task CAUSAL_LM \
    --do-inference-check true \
    --do-translation true \