# Machine Translation Fine-Tuning

## Overview
This repository provides scripts for fine-tuning machine translation models (Mistral-7B and mT5) on a custom dataset.
The fine-tuned models are tested and evaluated, with comparisons against reference translations from DeepL and MistralAI (Mistral Medium).

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/gri-sha/fine_tuning_mt.git
cd fine_tuning_mt
```

2. Set up the virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage Guide

### Fine-Tuning Models

1. Start a new screen session:
```bash
screen -S finetuning
source .venv/bin/activate
```

2. Make the script executable and run it:
```bash
chmod +x finetune/mistral7b/v3.sh
nohup ./finetune/mistral7b/v3.sh > sft_checkpoints/mistral7b/v3/output.log 2>&1 &
```

3. Monitor the logs:
```bash
tail -f sft_checkpoints/mistral7b/v3/output.log
```

4. Detach from the screen session with `Ctrl+A+D`

### Generating Translations

1. Start a new screen session:
```bash
screen -S translation
source .venv/bin/activate
```

2. Configure the translation parameters in `translate/translations_config.yml`

3. Run the translation script:
```bash
nohup python3 translate/main.py &
```

4. Detach from the screen session with `Ctrl+A+D`

### Evaluating Results

1. Configure the evaluation parameters in `evaluate/eval_config.yml`

2. Run the evaluation script:
```bash
python3 evaluate/main.py
```

## Notes
- For convinience it it better to run all the scripts from the **project root**
- For long-running processes (fine-tuning and translation), it's recommended to use `screen` or `tmux` to maintain sessions
- Check the model-specific shell scripts in `finetune/mistral7b/` and `finetune/mt5/` for additional configuration details
- Evaluation results are saved in `evaluate/evaluations.csv`