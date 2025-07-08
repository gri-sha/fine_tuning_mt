# %%
import os
import json
import random
import torch
from util import generate_simple_prompts, initialize_df, split_dataset, _config
from datasets import Dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from pprint import pprint

# %%
seed = 1212

# dataset splits
train_split = 0.85
test_split = 0.1
validation_split = 0.05

# directories
output_dir = "./finetuning_mistral7b_v1"
cache_dir = os.path.expanduser("~/.cache/huggingface/")
logs_path = os.path.join(output_dir, "logs.json")

# training
model_name = "mistralai/Mistral-7B-v0.1"
epochs = 1
learning_rate = 2e-3
batch_size = 32
max_seq_length = 512
logging_steps = 20

# %%
print("Dataset configuration :")
pprint(_config)

# %%
set_seed(seed)

# %%
df = initialize_df()

# %%
prompts = generate_simple_prompts(df, shots=0) + generate_simple_prompts(
    df, shots=1, fuzzy=True
)
random.shuffle(prompts)

# %%
print(type(prompts))
print(prompts[0], "\n")
print(prompts[-1])

# %%
dataset = Dataset.from_dict({"text": prompts})
dataset = split_dataset(
    dataset, train=train_split, test=test_split, validation=validation_split
)
pprint(dataset)

# %%
pprint(dataset["train"][4])

# %%
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# %%
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=nf4_config,
    use_cache=False,
    cache_dir=cache_dir,
)

# %%
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    add_bos_token=True,
    add_eos_token=False,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %%
peft_config = LoraConfig(
    lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# %%
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=0,
    logging_steps=logging_steps,
    save_strategy="epoch",
    # evaluation_strategy="epoch",
    learning_rate=learning_rate,
    bf16=True,
    lr_scheduler_type="constant",
)

# %%
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=0,
    logging_steps=logging_steps,
    save_strategy="epoch",
    # evaluation_strategy="epoch",
    learning_rate=learning_rate,
    bf16=True,
    lr_scheduler_type="constant",
    max_seq_length=max_seq_length,
    packing=True,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    # tokenizer=tokenizer,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# %%
tokenizer.decode(trainer.train_dataset["input_ids"][0])

# %%
trainer.train()

# %%
logs = trainer.state.log_history
with open(logs_path, "w") as log:
    log.write(json.dumps(logs, indent=2))
