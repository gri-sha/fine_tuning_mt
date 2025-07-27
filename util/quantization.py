import torch
from transformers import BitsAndBytesConfig
from .convert import str_to_bool


# Only 4 bit and 8 bit quantizations are supported by bitsandbytes
def get_bnb_config(args):
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if str_to_bool(args.bf16_for_compute) else None
            ),
            bnb_4bit_quant_type=args.quant_type,
            bnb_4bit_use_double_quant=str_to_bool(args.double_quant),
        )
    elif args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_has_fp16_weight=str_to_bool(args.llm_int8_has_fp16_weight),
            llm_int8_skip_modules=(
                [m.strip() for m in args.llm_int8_skip_modules.split(",")]
                if args.llm_int8_skip_modules
                else None
            ),
        )
    else:
        bnb_config = None  # No quantization
    return bnb_config
