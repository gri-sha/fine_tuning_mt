from typing import Optional
import argparse


def str2bool(value: str) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "n", "0"):
        return False
    elif value == "_":
        return None
    else:
        raise ValueError(f"Boolean value expected, got: {value}")


def _add_quantization_args(parser):
    quant_grp = parser.add_argument_group("Quantization")
    quant_grp.add_argument(
        "--quantization", type=str, default="", choices=["", "4bit", "8bit"]
    )
    quant_grp.add_argument("--double-quant", type="boolean", default=True)
    quant_grp.add_argument("--quant-type", type=str, choices=["nf4", "fp4"])
    quant_grp.add_argument("--bf16-for-compute", type="boolean", default=True)
    quant_grp.add_argument("--llm-int8-threshold", type=float, default=6.0)
    quant_grp.add_argument("--llm-int8-has-fp16-weight", type="boolean", default=False)
    quant_grp.add_argument("--llm-int8-skip-modules", nargs="*", type=str, default=None)


def _add_lora_args(parser):
    lora_grp = parser.add_argument_group("LoRA")
    lora_grp.add_argument("--lora-alpha", type=int, default=16)
    lora_grp.add_argument("--lora-dropout", type=float, default=0.1)
    lora_grp.add_argument("--lora-rank", type=int, default=64)
    lora_grp.add_argument("--lora-bias", type=str, default="none")
    lora_grp.add_argument("--lora-task", type=str, default="CAUSAL_LM")
    lora_grp.add_argument("--lora-targets", nargs="*", type=str, default=None)


def _add_tokenizer_args(parser):
    tok_grp = parser.add_argument_group("Tokenizer")
    tok_grp.add_argument("--add-bos-token", type="boolean", default=None)
    tok_grp.add_argument("--bos-token", type=str, default=None)
    tok_grp.add_argument("--add-eos-token", type="boolean", default=None)
    tok_grp.add_argument("--eos-token", type=str, default=None)
    tok_grp.add_argument(
        "--pad-side", type=str, default=None, choices=["left", "right"]
    )
    tok_grp.add_argument("--pad-token", type=str, default=None)
    tok_grp.add_argument("--use-fast-tokenizer", type="boolean", default=True)
    tok_grp.add_argument("--other-special-tokens", nargs="*", type=str, default=None)


def parse_training_arguments():
    parser = argparse.ArgumentParser()
    parser.register("type", "boolean", str2bool)

    # Model paramatres
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument("--model-name", type=str)
    model_grp.add_argument("--output-dir", type=str)

    # Data parameters
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument("--shots", nargs="+", type=int, default=[0])
    data_grp.add_argument("--fuzzy", nargs="+", type="boolean", default=[None])
    data_grp.add_argument("--val-shots", nargs="*", type=int, default=None)
    data_grp.add_argument("--val-fuzzy", nargs="*", type="boolean", default=None)

    # Training parameters
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--epochs", type=int)
    train_grp.add_argument("--learning-rate", type=float)
    train_grp.add_argument("--batch-size", type=int)
    train_grp.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_grp.add_argument("--packing", type="boolean", default=False)
    train_grp.add_argument("--max-seq-length", type=int, default=512)
    train_grp.add_argument("--logging-steps", type=int, default=20)
    train_grp.add_argument("--completion-only-loss", type="boolean", default=False)
    train_grp.add_argument("--warmup-steps", type=int, default=0)
    train_grp.add_argument(
        "--eval-strategy", type=str, default="epoch", choices=["epoch", "steps", "no"]
    )
    train_grp.add_argument("--eval-steps", type=int, default=0)
    train_grp.add_argument(
        "--save-strategy", type=str, default="epoch", choices=["epoch", "steps"]
    )
    train_grp.add_argument("--save-steps", type=int, default=0)

    # Runtime parameters
    run_grp = parser.add_argument_group("Runtime")
    run_grp.add_argument("--do-trainer-check", type="boolean", default=True)
    run_grp.add_argument("--do-training", type="boolean", default=True)
    run_grp.add_argument("--do-evaluation", type="boolean", default=True)

    # Common parametres
    _add_quantization_args(parser)
    _add_lora_args(parser)
    _add_tokenizer_args(parser)

    return parser.parse_args()


def parse_translation_arguments():
    parser = argparse.ArgumentParser()
    parser.register("type", "boolean", str2bool)

    # Model paramatres
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument("--checkpoint-path", type=str)
    model_grp.add_argument("--translations-path", type=str)
    model_grp.add_argument("--use-base-model", type="boolean", default=False)
    model_grp.add_argument("--model-name", type=str, default=None)
    model_grp.add_argument("--max-new-tokens", type=int, default=128)

    # Data parameters
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument("--batch-size", type=int, default=16)
    data_grp.add_argument("--use-batches", type="boolean", default=True)
    data_grp.add_argument("--shots", nargs="+", type=int, default=[0])
    data_grp.add_argument("--fuzzy", nargs="+", type="boolean", default=[None])

    # Runtime parameters
    run_grp = parser.add_argument_group("Runtime")
    run_grp.add_argument("--do-inference-check", type="boolean", default=True)
    run_grp.add_argument("--do-translation", type="boolean", default=True)

    # Common parametres
    _add_quantization_args(parser)
    _add_lora_args(parser)
    _add_tokenizer_args(parser)

    return parser.parse_args()


def parse_evaluation_arguments():
    parser = argparse.ArgumentParser()
    parser.register("type", "boolean", str2bool)

    parser.add_argument("--tr-dir", type=str)
    parser.add_argument("--eval-path", type=str, default="results/evaluations.csv")

    return parser.parse_args()
