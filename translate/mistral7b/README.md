# Mistral-7B-v0.3 Translation Configuration

## Overview
This configuration ensures consistent tokenizer behavior between training and inference phases for Mistral-7B-v0.3.

## Tokenizer Settings
- `add_eos_token`: `false` (In order not to stop model generation)
- `add_bos_token`: `true` (BOS token is applicable and should be added)
- **Fast Tokenizer**: Supported and can be used for improved performance

## Token Configuration
- **BOS Token**: `<s>` 
- **EOS Token**: - 
- **PAD Token**: `</s>`
  
## Padding Strategy
- **Training**: Right padding
- **Inference**: Left padding
