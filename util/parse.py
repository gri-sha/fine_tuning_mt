import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--output-dir", type=str)

    # Data parameters
    parser.add_argument("--shots", type=str, default="0")
    parser.add_argument("--fuzzy", type=str, default="false")
    parser.add_argument("--bos_token", type=str, default="true")
    parser.add_argument("--eos-token", type=str, default="false")
    parser.add_argument("--pad-side", type=str, default="right")
    
    # Training parameters
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--packing", type=str, default="true")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--completion-only-loss", type=str, default="")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--eval-steps", type=str, default="")
    
    # LoRA parameters
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-bias", type=str, default="none")
    parser.add_argument("--lora-task", type=str, default="CAUSAL_LM")

    return parser.parse_args()
