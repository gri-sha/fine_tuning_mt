# Mistral-7B-v0.3 Finetuning Configuration

## Overview
Configuration notes for the Mistral-7B-v0.3 decoder-only model tokenizer setup.

## Key Configuration Details

### Model Architecture
- **Model Type**: Decoder-only architecture
- **Packing**: Not used due to compatibility issues between Flash Attention 2 and current torch/CUDA versions

### Tokenizer Settings
- `add_eos_token`: `false` (SFTTrainer automatically handles EOS token addition, when LlamaTokenizer(Fast) is used)
- `add_bos_token`: `true` (BOS token is applicable and should be added)
- **Fast Tokenizer**: Supported and can be used for improved performance

### Token Configuration
- **PAD Token**: `</s>` (same as EOS token)
- **BOS Token**: `<s>`
- **EOS Token**: `</s>`

### Padding Configuration
- **Training**: Right-side padding
- **Inference**: Left-side padding


## Notes
The tokenizer maintains consistency by using the same token (`</s>`) for both padding and end-of-sequence marking. The automatic EOS token handling by SFTTrainer eliminates the need for manual EOS token addition during training.