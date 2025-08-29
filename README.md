# Machine Translation Fine-Tuning

[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.53.2-orange)](https://huggingface.co/transformers/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/🤗%20PEFT-0.16.0-yellow)](https://github.com/huggingface/peft)
[![TRL](https://img.shields.io/badge/🤗%20TRL-0.19.1-purple)](https://github.com/huggingface/trl)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-0.46.1-green)](https://github.com/TimDettmers/bitsandbytes)

[![DeepL API](https://img.shields.io/badge/DeepL-API-0F2B46?logo=deepl&logoColor=white)](https://www.deepl.com/pro-api)
[![Mistral AI](https://img.shields.io/badge/Mistral-AI-FF7000?logo=mistral-ai&logoColor=white)](https://mistral.ai/)


## Overview

This study evaluates whether specialized fine-tuning of smaller models can achieve competitive translation quality compared to large commercial APIs.
The research compares four fine-tuned models against two industry-standard baselines on a custom dataset.

*Fine-tuned Models:*
- **Mistral-7B-v0.3**
- **mT5** (xl)
- **Qwen3-14B**
- **UmT5** (xl)

*Baseline Comparisons:*
- **Mistral Medium** (via API)
- **DeepL** (commercial translation service)


## Installation

```bash
git clone https://github.com/gri-sha/fine_tuning_mt.git && cd fine_tuning_mt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

### Model Fine-Tuning
Execute model-specific training configurations:
```bash
bash finetune/<model>/<version>.sh
```

Training scripts contain hyperparameter configurations and dataset specifications. Utilize `nohup` for long-running training processes.

### Translation Generation

**Fine-tuned models:**
```bash
bash translate/<model>/<version>.sh
```

**APIs:**
```bash
python3 translate/api/<name>/run.py
```

### Evaluation

1. Configure evaluation parameters in `evaluate/eval_config.yml`:

2. Execute evaluation:
```bash
python3 evaluate/run.py
```

Results are exported to `results/evaluation.csv`.
Performance metrics used: BLEU, chrF++, TER, COMET.

## References

- Moslem et al. (2023). [Adaptive Machine Translation with LLMs](https://doi.org/10.48550/arXiv.2301.13294)
- Moslem et al. (2023). [Fine-tuning LLMs for Adaptive MT](https://doi.org/10.48550/arXiv.2312.12740)

## Implementation

- Based on: [ymoslem/Adaptive-MT-LLM-Fine-tuning](https://github.com/ymoslem/Adaptive-MT-LLM-Fine-tuning)
