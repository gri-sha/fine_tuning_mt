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
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
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
    SEED,
    TEST_SPLIT,
    VALID_SPLIT,
)

# Load arguments
args = parse_arguments()

output_dir = args.output_dir
cache_dir = os.path.expanduser("~/.cache/huggingface/")
logs_path = os.path.join(output_dir, "logs.json")

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Print configs
print("\nDataframe configuration:")
pprint(_config)

print("Training Configuration:")
pprint(args.__dict__)

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

shots = args.shots
fuzzy = args.fuzzy

if len(shots) != len(fuzzy):
    raise ValueError("The number of 'shots' must match the number of 'fuzzy' values.")

for s, f in zip(shots, fuzzy):
    if f is None and s != 0:
        raise ValueError("If 'fuzzy' is None, 'shots' must be 0.")
    _, p, c = generate_instruction_prompts(df_train, shots=s, fuzzy=f)
    prompts.extend(p)
    completions.extend(c)

dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
dataset = validation_split(dataset, validation=VALID_SPLIT)
pprint(dataset)

quant_config = None
if args.quantization == "4bit":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if args.bf16_for_compute else None
        ),
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=args.double_quant,
    )
elif args.quantization == "8bit":
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=args.llm_int8_threshold,
        llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
        llm_int8_skip_modules=args.llm_int8_skip_modules
    )

# if there is an error, that directories related to cuda are not found, the easiest solution is to reinstall bitsandbytes
if args.lora_task == "CAUSAL_LM":
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quant_config,
        use_cache=False,
        cache_dir=cache_dir,
        # attn_implementation='flash_attention_2'  # not compatible with current versions of torch and cuda
    )
elif args.lora_task == "SEQ_2_SEQ_LM":
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quant_config,
        use_cache=False,
        cache_dir=cache_dir,
        # attn_implementation='flash_attention_2'  # not compatible with current versions of torch and cuda
    )
else:
    raise ValueError("LoRA task type is not supported yet.")

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    cache_dir=cache_dir,
    add_bos_token=args.bos_token,
    add_eos_token=args.eos_token,
    legacy=False,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = args.pad_side

peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_rank,
    bias=args.lora_bias,
    task_type=args.lora_task,
    target_modules=args.lora_targets,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print(f"\nModel device: {next(model.parameters()).device}", "\n")

# Setup training
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=args.warmup_steps,
    logging_steps=args.logging_steps,
    do_train=True,
    save_strategy=args.save_strategy,
    save_steps=args.save_steps if args.save_steps else None,
    do_eval=True,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps if args.eval_steps else None,
    learning_rate=args.learning_rate,
    bf16=args.bf16_for_compute,
    lr_scheduler_type="constant",
    max_seq_length=args.max_seq_length,
    packing=args.packing,
    completion_only_loss=args.completion_only_loss,
    dataset_kwargs={"add_special_tokens": False},
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# for _ in range(4):
#     print(tokenizer.decode(trainer.train_dataset["input_ids"][_], skip_special_tokens=False), "\n")
# sys.exit(0)

# Start training
trainer.train()

# Save training logs
logs = trainer.state.log_history
with open(logs_path, "w") as log_file:
    json.dump(logs, log_file, indent=2)
