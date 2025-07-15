## Fine-tuning Methodology

The fine-tuning approach in this implementation follows the methodologies established in:

- **[Moslem et al. (2023)](https://doi.org/10.48550/arXiv.2301.13294)**: *Adaptive Machine Translation with Large Language Models*
- **[Moslem et al. (2023)](https://doi.org/10.48550/arXiv.2312.12740)**: *Fine-tuning Large Language Models for Adaptive Machine Translation*

## Key Implementation Details

- **No special formatting tokens**: The training data does not use format-specific tokens such as `<s>`, `</s>`, `<INST>`, or `</INST>`
- **Unified format**: Instructions and responses are presented as continuous text without explicit delimiters or segmentation markers

## Code Repository
The implementation is based on the open-source code available at: [ymoslem/Adaptive-MT-LLM-Fine-tuning](https://github.com/ymoslem/Adaptive-MT-LLM-Fine-tuning)
