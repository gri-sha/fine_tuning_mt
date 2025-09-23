import os
import sys
import json
import shutil
import random
import itertools
from pprint import pprint
from torch import where, any
from transformers import TrainerCallback
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
    get_quant_config,
    get_peft_config,
    load_model,
    load_tokenizer,
    check_cuda_availability,
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
check_cuda_availability()

# Load data
dataset = create_train_dataset(
    shots=args.shots,
    fuzzy=args.fuzzy,
    val_shots=args.val_shots if args.val_shots is not None else args.shots,
    val_fuzzy=args.val_fuzzy if args.val_fuzzy is not None else args.fuzzy,
)
pprint(dataset)

# Setup quantization
quant_config = get_quant_config(args)
pprint(quant_config)

# LoRA setup
peft_config = get_peft_config(args)
pprint(peft_config)

# Load model and tokenizer
# if there is an error, that directories related to cuda are not found, the easiest solution is to reinstall bitsandbytes
model = load_model(args, quant_config, peft_config, cache_dir, inference=False)
tokenizer = load_tokenizer(args, cache_dir)

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

# This callback saves the model checkpoint with least eval loss
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, save_dir, trainer):
        self.best_loss = float("inf")
        self.save_dir = save_dir
        self.trainer = trainer
        self.last_checkpoint_path = None
        os.makedirs(self.save_dir, exist_ok=True)
    
    # This method is called after evaluation 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_loss = metrics.get("eval_loss")
        print("Evaluation report:")
        print(f"Current eval_loss: {eval_loss}")
        
        if eval_loss is not None and eval_loss < self.best_loss:
            print(f"New best loss! Previous: {self.best_loss:.6f}, New: {eval_loss:.6f}")
            self.best_loss = eval_loss
            
            # Delete previous checkpoint
            if self.last_checkpoint_path and os.path.exists(self.last_checkpoint_path):
                print(f"Removing previous checkpoint: {self.last_checkpoint_path}")
                shutil.rmtree(self.last_checkpoint_path)
            
            # Save model
            checkpoint_name = f"best_loss_step_{state.global_step}"
            checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
            self.last_checkpoint_path = checkpoint_path
            
            print(f"Saving new best model to: {checkpoint_path}")
            self.trainer.save_model(checkpoint_path)
        else:
            if eval_loss is not None:
                print(f"Loss {eval_loss:.6f} >= best loss {self.best_loss:.6f}, not saving")

        return control
trainer.add_callback(SaveBestModelCallback(output_dir, trainer))

# Tokenization check
if args.do_trainer_check:
    # NOTE: The following section displays the configuration provided to SFTTrainer.
    # (regardless the 'processing_class' is used)
    # Be aware that these settings may be modified internally by the trainer.
    # The trainer may silently override these settings without any warnings.
    # Verify the tokenization examples below for correctness.
    print(f"\nTokenizer: {trainer.processing_class.__class__.__name__}")
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
            eos_positions = where(
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
                if any(rest_mask != 0):
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
