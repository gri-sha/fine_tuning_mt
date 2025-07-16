# Machine Translation Fine-tuning of Mistral-7B-v0.1

## Versions

### V1 & V2
- **Format**: Language modeling
- **Features**: Single "text" feature in dataset (prompt and completion are concatenated)
- **Tokens**: No special formatting tokens (`<s>`, `</s>`, `<INST>`, `</INST>`)
- **Structure**: Unified format without explicit delimiters

**Note**: This format is **unsupported** by SFTTrainer but matches the original research implementation.

### V3
- **Format**: Instruction
- **Features**: `{"prompt": "<prompt text>", "completion": "<ideal generated text>"}`
- **Compatibility**: Fully supported by SFTTrainer (TRL 0.19.1 and 0.7.10)
- **Performance**: Expected to yield better results

## References

- Moslem et al. (2023). [Adaptive Machine Translation with Large Language Models](https://doi.org/10.48550/arXiv.2301.13294)
- Moslem et al. (2023). [Fine-tuning Large Language Models for Adaptive Machine Translation](https://doi.org/10.48550/arXiv.2312.12740)

## Implementation

Based on: [ymoslem/Adaptive-MT-LLM-Fine-tuning](https://github.com/ymoslem/Adaptive-MT-LLM-Fine-tuning)