from torch import bfloat16
from torch.cuda import is_available, get_device_name
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def get_quant_config(args):
    quant_config = None
    if args.quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(bfloat16 if args.bf16_for_compute else None),
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
    return quant_config


def get_peft_config(args):
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_rank,
        bias=args.lora_bias,
        task_type=args.lora_task,
        target_modules=args.lora_targets,
    )
    return peft_config


def load_model(args, quant_config, peft_config, cache_dir, inference=False):
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

    if inference:
        model = get_peft_model(model, peft_config)
    else:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    print("Model loaded.", "\n")
    print(f"\nModel device: {next(model.parameters()).device}", "\n")

    return model


def load_tokenizer(args, cache_dir, model_name=None):
    print(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name if args.model_name is not None else model_name,
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

    print("Tokenizer loaded.")
    return tokenizer


def check_cuda_availability():
    if not is_available():
        print("\nWarning: CUDA is not available.", "\n")
    else:
        print(f"\nCUDA device: {get_device_name(0)}", "\n")
