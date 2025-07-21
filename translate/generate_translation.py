import os
import yaml
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

def generate_response(prompt, model):
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to("cuda")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=tr_config["max_new_tokens"],
        min_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_output = tokenizer.batch_decode(generated_ids)
    return decoded_output[0].replace(prompt, "")

pprint(dataset["prompts"][35])

response = generate_response(dataset["prompts"][35], model)
print(f"Reference: {dataset['references'][35]}")
print(f"Response: {response}")
