import os
import sys
import json
import random
import itertools
from pathlib import Path
from pprint import pprint
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util import (
    create_train_dataset,
    login_to_hf,
    parse_training_arguments,
    plot_log_metrics,
    dataset_config,
)

# Load arguments
args = parse_training_arguments()

output_dir = args.output_dir
cache_dir = os.path.expanduser("~/.cache/huggingface/")
logs_path = os.path.join(output_dir, "logs.json")
training_graph_path = os.path.join(output_dir, "training_graph.png")

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Print dataset configuration
print("\nDataset processing configuration:")
pprint(dataset_config)

print("\nTraining Configuration:")
pprint(args.__dict__)

# Login to Hugging Face
login_to_hf()

# Check CUDA availability
if not torch.cuda.is_available():
    print("\nWarning: CUDA is not available.", "\n")
else:
    print(f"\nCUDA device: {torch.cuda.get_device_name(0)}", "\n")

# Load data
dataset = create_train_dataset(args.shots, args.fuzzy)
pprint(dataset)

# Setup quantization
quant_config = None
if args.quantization == "4bit":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if args.bf16_for_compute else None),
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=args.double_quant,
    )
elif args.quantization == "8bit":
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=args.llm_int8_threshold,
        llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
        llm_int8_skip_modules=args.llm_int8_skip_modules,
    )

# Setup LoRA
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

# Setup peft configuration
peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_rank,
    bias=args.lora_bias,
    task_type=args.lora_task,
    target_modules=args.lora_targets,
)

# Load model
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print(f"\nModel device: {next(model.parameters()).device}", "\n")


# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    cache_dir=cache_dir,
    legacy=False,
    use_fast=args.use_fast_tokenizer,
)

if args.add_bos_token is not None:
    tokenizer.add_bos_token = args.add_bos_token
if args.bos_token is not None:
    tokenizer.bos_token = args.bos_token
if args.eos_token is not None:
    tokenizer.eos_token = args.eos_token
if args.add_eos_token is not None:
    tokenizer.add_eos_token = args.add_eos_token
if args.pad_token is not None:
    tokenizer.pad_token = args.pad_token
if args.pad_side:
    tokenizer.padding_side = args.pad_side

# Setup training
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    eval_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=args.warmup_steps,
    logging_steps=args.logging_steps,
    do_train=True,
    save_strategy=args.save_strategy,
    save_steps=args.save_steps if args.save_steps else None,
    do_eval=args.do_evaluation,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps if args.eval_steps else None,
    learning_rate=args.learning_rate,
    bf16=args.bf16_for_compute,
    lr_scheduler_type="constant",
    max_seq_length=args.max_seq_length,
    packing=args.packing,
    completion_only_loss=args.completion_only_loss,
    label_names=["labels"],
)

pprint(sft_config)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Tokenization check
if args.do_tokenizer_check:
    # NOTE: The following section displays the configuration provided to SFTTrainer.
    # Be aware that these settings may be modified internally by the trainer.
    # The trainer may silently override these settings without any warnings.
    # Verify the tokenization examples below for correctness.
    print(f"\nTokenizer: {tokenizer.__class__.__name__}")
    try:
        print(
            "BOS token:",
            repr(trainer.processing_class.bos_token),
            "enabled" if trainer.processing_class.add_bos_token else "disabled",
        )
    except AttributeError:
        print("BOS token isn't applicable for this tokenizer")
    try:
        print(
            "EOS token:",
            repr(trainer.processing_class.eos_token),
            "enabled" if trainer.processing_class.add_eos_token else "disabled",
        )
    except AttributeError:
        print("EOS token isn't applicable for this tokenizer")
    print("PAD token:", repr(trainer.processing_class.pad_token))

    print("\nAll special tokens:")
    pprint(trainer.processing_class.all_special_tokens)

    if args.packing:
        print("\nChecking tokenized sentences...")
        idx = random.choice(range(len(trainer.train_dataset)))
        print(f"Decoded input {idx}:")
        print(
            repr(
                trainer.processing_class.decode(trainer.train_dataset["input_ids"][idx])
            )
        )
        print("Positional ids:")
        print(trainer.train_dataset["position_ids"][idx])
        print("Total tokens:", len(trainer.train_dataset["input_ids"][idx]))

        print("\nChecking batches from tokenized dataset...")
        train_dataloader = trainer.get_train_dataloader()
        idx = random.randint(0, len(train_dataloader) - 1)
        batch = next(itertools.islice(train_dataloader, idx, idx + 1))

        print("\nBatch:", idx)
        print("\nKeys:", batch.keys())
        print("\nTensor shapes:", {k: v.shape for k, v in batch.items()})

        decoded_inputs = trainer.processing_class.batch_decode(
            batch["input_ids"], skip_special_tokens=False
        )
        print("\nDecoded batch:")
        print(repr(decoded_inputs[0]))
        print("Attention mask:")
        print(batch["attention_mask"][0])

    else:
        NUM_BATCH_SENTENCES = 3
        NUM_REG_SENTENCES = 3

        print("\nChecking tokenized sentences...")
        idxs = random.sample(range(len(trainer.train_dataset)), NUM_REG_SENTENCES)
        for i in idxs:
            print(f"Decoded input {i}:")
            print(
                repr(
                    trainer.processing_class.decode(
                        trainer.train_dataset["input_ids"][i]
                    )
                )
            )
            print("Completion mask:")
            print(trainer.train_dataset["completion_mask"][i])

        print("\nChecking batches from tokenized dataset...")
        train_dataloader = trainer.get_train_dataloader()
        idx = random.randint(0, len(train_dataloader) - 1)
        batch = next(itertools.islice(train_dataloader, idx, idx + 1))

        print("\nBatch:", idx)
        print("\nKeys:", batch.keys())
        print("\nTensor shapes:", {k: v.shape for k, v in batch.items()})

        decoded_inputs = trainer.processing_class.batch_decode(
            batch["input_ids"], skip_special_tokens=False
        )
        batch_idxs = random.sample(range(len(decoded_inputs)), NUM_BATCH_SENTENCES)
        for i in batch_idxs:
            print(f"\nDecoded input {i}:")
            print(repr(decoded_inputs[i]))
            print("Attention mask:")
            pprint(batch["attention_mask"][i])
            # find the eos tokens
            eos_positions = torch.where(
                batch["input_ids"][i] == trainer.processing_class.eos_token_id
            )[0].tolist()
            if eos_positions:
                print("EOS token positions:", eos_positions)
            else:
                print("No EOS token found in this input.")
            print(
                "Attention mask at 1st EOS position:",
                batch["attention_mask"][i][eos_positions[0]].item(),
            )
            if eos_positions[0] < len(batch["attention_mask"][i]) - 1:
                # Check all tokens after the first EOS position
                rest_mask = batch["attention_mask"][i][eos_positions[0] + 1 :]
                if torch.any(rest_mask != 0):
                    print("Warning: Attention mask after EOS contains non-zero values!")
                    print("Attention mask after EOS:", rest_mask.tolist())
                else:
                    print("All attention mask values after 1st EOS are 0 as expected.")
            else:
                print("EOS token is the last token in this input.")

# Start training
if args.do_training:
    trainer.train()

    # Save training logs
    logs = trainer.state.log_history
    with open(logs_path, "w") as log_file:
        json.dump(logs, log_file, indent=2)

    # Plot metrics from the logs
    plot_log_metrics(logs, plot_path=training_graph_path)
