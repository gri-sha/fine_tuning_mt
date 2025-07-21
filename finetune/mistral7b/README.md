# Machine Translation Fine-tuning of Mistral-7B-v0.1

## Versions

### V1 & V2
- **Format**: Language modeling
- **Features**: Single "text" feature in dataset (prompt and completion are concatenated), without special formatting tokens (`<s>`, `</s>`, `<INST>`, `</INST>`)

### V3
- **Format**: Instruction
- **Features**: `{"prompt": "<prompt text>", "completion": "<ideal generated text>"}`

## References

- Moslem et al. (2023). [Adaptive Machine Translation with Large Language Models](https://doi.org/10.48550/arXiv.2301.13294)
- Moslem et al. (2023). [Fine-tuning Large Language Models for Adaptive Machine Translation](https://doi.org/10.48550/arXiv.2312.12740)

## Implementation

Based on: [ymoslem/Adaptive-MT-LLM-Fine-tuning](https://github.com/ymoslem/Adaptive-MT-LLM-Fine-tuning)