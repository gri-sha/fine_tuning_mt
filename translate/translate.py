import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from pprint import pprint
from util import generate_instruction_prompts, initialize_dfs

with open("translate/translations_config.yml", "r") as f:
    tr_config = yaml.safe_load(f)

cache_dir = os.path.expanduser("~/.cache/huggingface/")

set_seed(tr_config["seed"])

_, df_test = initialize_dfs(test=tr_config["test_split"])
s0, p0, r0 = generate_instruction_prompts(df_test, shots=0)
s1, p1, r1 = generate_instruction_prompts(df_test, shots=1, fuzzy=True)
sources = s0 + s1
references = r0 + r1
prompts = p0 + p1
dataset = Dataset.from_dict(
    {"sources": sources, "references": references, "prompts": prompts}
)
pprint(dataset)


peftconfig = PeftConfig.from_pretrained(tr_config["model_checkpoint"])

model_base = AutoModelForCausalLM.from_pretrained(
    peftconfig.base_model_name_or_path, device_map="auto", cache_dir=cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(
    tr_config["model_name"],
    cache_dir=cache_dir,
    add_bos_token=tr_config["bos_token"],
    add_eos_token=tr_config["eos_token"],  # always False for inference
)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(model_base, tr_config["model_checkpoint"])
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


os.makedirs(tr_config["translations_dir"], exist_ok=True)

translations_df = pd.DataFrame(
    {"sources": sources, "references": references, "translations": translations}
)
translations_df.to_csv(os.path.join(tr_config["translations_dir"], tr_config["translations_name"]), index=False)
