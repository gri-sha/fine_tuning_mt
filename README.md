# Machine Translation Fine-Tuning
Comparison on on custom dataset of finetuned smaller models against popular LLMs.

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