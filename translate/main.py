import os
import sys
import yaml
import json
import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import set_seed, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
from pprint import pprint

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util import generate_instruction_prompts, initialize_dfs, SEED, TEST_SPLIT

with open("translate/translations_config.yml", "r") as f:
    tr_config = yaml.safe_load(f)

with open(os.path.join(tr_config["checkpoint_path"], "tokenizer_config.json"), "r") as f:
    tok_config = json.load(f)

cache_dir = os.path.expanduser("~/.cache/huggingface/")

# we set the seed not for splits but for model initialization
# test split is done by index
set_seed(SEED)  

_, df_test = initialize_dfs(test=TEST_SPLIT)
s0, p0, r0 = generate_instruction_prompts(df_test, shots=0)
s1, p1, r1 = generate_instruction_prompts(df_test, shots=1, fuzzy=True)
sources = s0 + s1
references = r0 + r1
prompts = p0 + p1
dataset = Dataset.from_dict(
    {"sources": sources, "references": references, "prompts": prompts}
)
pprint(dataset)

peftconfig = PeftConfig.from_pretrained(tr_config["checkpoint_path"])

if peftconfig.task_type == "SEQ_2_SEQ_LM":
    model_base = AutoModelForSeq2SeqLM.from_pretrained(
        peftconfig.base_model_name_or_path, device_map="auto", cache_dir=cache_dir
    )
elif peftconfig.task_type == "CAUSAL_LM":
    model_base = AutoModelForCausalLM.from_pretrained(
        peftconfig.base_model_name_or_path, device_map="auto", cache_dir=cache_dir
    )
else: 
    raise ValueError(f"Unknown model task: {peftconfig.task_type}")

tokenizer = AutoTokenizer.from_pretrained(
    peftconfig.base_model_name_or_path,
    cache_dir=cache_dir,
    add_bos_token=tok_config["add_bos_token"],
    add_eos_token=tok_config["add_eos_token"],
    legacy=False
)

tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(model_base, tr_config["checkpoint_path"])
print("Peft model loaded")


def generate_batch_responses(prompts, model):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        "cuda"
    )
    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=tr_config["max_new_tokens"],
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


translations = []
for i in tqdm(range(0, len(dataset), tr_config["batch_size"])):
    batch_prompts = dataset["prompts"][i : i + tr_config["batch_size"]]
    responses = generate_batch_responses(batch_prompts, model)
    cleaned = [r.replace(p, "") for p, r in zip(batch_prompts, responses)]
    translations.extend(cleaned)

directory, _ = os.path.split(tr_config["translations_path"])
os.makedirs(directory, exist_ok=True)

translations_df = pd.DataFrame(
    {"sources": sources, "references": references, "translations": translations}
)
translations_df.to_csv(tr_config["translations_path"], index=False)
