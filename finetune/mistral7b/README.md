## Machine Translation Fine-tuning of Mistral-7B

| Version | Base Model | Comp. Only Loss | Quant. | Epochs | Shots Included | Packing | Batch Size | Learning Rate | Notes |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **V1** | v0.1 | False | 4-bit | 1 | 0, 1 | Yes | 32 | 1e-4 | <ul><li>Poor results</li><li>FlashAttention 2 incompatible</li></ul> |
| **V2** | v0.1 | False | 4-bit | 1 | 0, 1 | Yes | 32 | 2e-3 | <ul><li>Poor results</li><li>FlashAttention 2 incompatible</li></ul> |
| **V3** | v0.1 | True | 4-bit | 2 | 0, 1 | No | 64 | 2e-3 | <ul><li>Poor results</li><li>FlashAttention 2 incompatible</li><li>More validation</li></ul> |
| **V4** | v0.1 | True | 4-bit | 3 | 0, 1, 2 | No | 64 | 2e-3 | <ul><li>Proper setup</li></ul> |
| **V5** | v0.3 | True | 4-bit | 3 | 0, 1, 2 | No | 64 | 2e-4 | <ul><li>Updated model version</li></ul> |

## References

* Moslem et al. (2023). [Adaptive Machine Translation with LLMs](https://doi.org/10.48550/arXiv.2301.13294)
* Moslem et al. (2023). [Fine-tuning LLMs for Adaptive MT](https://doi.org/10.48550/arXiv.2312.12740)

## Implementation

* Based on: [ymoslem/Adaptive-MT-LLM-Fine-tuning](https://github.com/ymoslem/Adaptive-MT-LLM-Fine-tuning)
