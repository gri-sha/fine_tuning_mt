from typing import Optional
import argparse

def str2bool(value: str) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    elif value == '_':
        return None
    else:
        raise ValueError(f"Boolean value expected, got: {value}")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.register('type', 'boolean', str2bool)

    # Model configuration
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--output-dir", type=str)

    # Data parameters
    parser.add_argument("--shots", nargs='+', type=int, default=[0])
    parser.add_argument("--fuzzy", nargs='+', type='boolean', default=[None])
    parser.add_argument("--bos-token", type='boolean', default=True)
    parser.add_argument("--eos-token", type='boolean', default=False)
    parser.add_argument("--pad-side", type=str, default="right")

    # Quantization parameters
    parser.add_argument("--quantization", type=str, default="", choices=["", "4bit", "8bit"])
    parser.add_argument("--double-quant", type='boolean', default=True)
    parser.add_argument("--quant-type", type=str, choices=["nf4", "fp4"])
    parser.add_argument("--llm-int8-threshold", type=float, default=6.0)
    parser.add_argument("--llm-int8-has-fp16-weight", type='boolean', default=False)
    parser.add_argument("--llm-int8-skip-modules", nargs='*', type=str, default=None)

    # LoRA parameters
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-bias", type=str, default="none")
    parser.add_argument("--lora-task", type=str, default="CAUSAL_LM")
    parser.add_argument("--lora-targets", nargs='*', type=str, default=None)

    # Training parameters
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--packing", type='boolean', default=False)
    parser.add_argument("--bf16-for-compute", type='boolean', default=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--completion-only-loss", type='boolean', default=False)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-strategy", type=str, default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save-steps", type=int, default=0)

    return parser.parse_args()
