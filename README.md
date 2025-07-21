# Machine Translation Fine-Tuning

## Overview
This repository provides scripts for fine-tuning machine translation models (Mistral-7B and mT5) on a custom dataset.
The fine-tuned models are tested and evaluated, with comparisons against reference translations from DeepL and ChatGPT.

## Repository Structure
```
.
├── data/                        # Dataset files
│   ├── all_en.txt
│   ├── all_fr.txt
│   ├── all.csv 
│   └── README.md 
│
├── finetune/                    
│   ├── finetuning.py            # Fine-tuning core logic
│   ├── mistral7b/               # Mistral-7B specific scripts
│   │   ├── README.md
│   │   ├── v1.sh           
│   │   ├── v2.sh           
│   │   └── v3.sh           
│   └── mt5/                     # mT5 specific scripts
│       ├── README.md
│       ├── v1.sh           
│       └── v2.sh           
│
├── evaluate/
│   ├── eval_config.yml          
│   ├── evaluate.py              # Main evaluation script
│   └── evaluations.csv          
│
├── translate/
│   ├── generate_translation.py
│   ├── translate_with_deepl.py
│   ├── translate.py             # Main translation script
│   ├── translations/            
│   │   ├── mistral7b_v1_translations.csv
│   │   ├── mistral7b_v2_translations.csv
│   │   ├── mistral7b_v3_translations.csv
│   │   ├── mt5_v1_translations.csv
│   │   └── mt5_v2_translations.csv
│   └── translations_config.yml
│
├── util/
│   ├── __init__.py
│   ├── convert.py
│   ├── dataset_config.yml
│   ├── fuzzy_matches.py
│   ├── login.py
│   ├── parse.py
│   ├── prompts.py
│   └── read_data.py
│
├── README.md
└── requirements.txt
```

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
nohup python3 translate/translate.py &
```

4. Detach from the screen session with `Ctrl+A+D`

### Evaluating Results

1. Configure the evaluation parameters in `evaluate/eval_config.yml`

2. Run the evaluation script:
```bash
python3 evaluate/evaluate.py
```

## Notes
- For long-running processes, it's recommended to use `screen` or `tmux` to maintain sessions
- Check the model-specific READMEs in `finetune/mistral7b/` and `finetune/mt5/` for additional configuration details
- Evaluation results will be saved in `evaluate/evaluations.csv`
- 