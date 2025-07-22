import os
import sys
import json
from pathlib import Path
from pprint import pprint
import torch
from datasets import Dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util import (
    generate_instruction_prompts,
    initialize_dfs,
    validation_split,
    login_to_hf,
    _config,
    parse_arguments,
    str_to_bool,
    read_shots,
    read_fuzzy,
    SEED,
    TEST_SPLIT,
    VALID_SPLIT,
)

args = parse_arguments()

output_dir = args.output_dir
cache_dir = os.path.expanduser("~/.cache/huggingface/")
logs_path = os.path.join(output_dir, "logs.json")

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Print configuration
print("Training Configuration:")
pprint(args.__dict__)

# Print dataframe configuration
print("\nDataframe configuration:")
pprint(_config)

# Login to Hugging Face
login_to_hf()

# Check CUDA availability
if not torch.cuda.is_available():
    print("\nWarning: CUDA is not available.", "\n")
else:
    print(f"\nCUDA device: {torch.cuda.get_device_name(0)}", "\n")

# Set seed
set_seed(SEED)

# Load data
df_train, _ = initialize_dfs(test=TEST_SPLIT)
prompts = []
completions = []

shots = read_shots(args.shots)
fuzzy = read_fuzzy(args.fuzzy)
for s, f in zip(shots, fuzzy):
    _, p, c = generate_instruction_prompts(df_train, shots=s, fuzzy=f)
    prompts.extend(p)
    completions.extend(c)

dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
dataset = validation_split(dataset, validation=VALID_SPLIT)
pprint(dataset)

# Load and setup model
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# if there is an error, that directories related to cuda are not found, the easiest solution is to reinstall bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    quantization_config=nf4_config,
    use_cache=False,
    cache_dir=cache_dir,
    # attn_implementation='flash_attention_2'  # not compatible with current versions of torch and cuda
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    cache_dir=cache_dir,
    add_bos_token=str_to_bool(args.bos_token),
    add_eos_token=str_to_bool(args.bos_token),
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = args.padding_side

peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_rank,
    bias=args.lora_bias,
    task_type=args.lora_task,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print(f"\nModel device: {next(model.parameters()).device}", "\n")

# Setup training
eval_steps = None
if args.eval_steps and args.eval_steps.strip():
    eval_steps = int(args.eval_steps)

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=args.warmup_steps,
    logging_steps=args.logging_steps,
    do_train=True,
    save_strategy="epoch",
    do_eval=True,
    eval_strategy=args.eval_strategy,
    eval_steps=eval_steps,
    learning_rate=args.learning_rate,
    bf16=True,
    lr_scheduler_type="constant",
    max_seq_length=args.max_seq_length,
    packing=str_to_bool(args.packing),
    completion_only_loss=str_to_bool(args.completion_only_loss),
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Start training
trainer.train()

# Save training logs
logs = trainer.state.log_history
with open(logs_path, "w") as log_file:
    json.dump(logs, log_file, indent=2)
