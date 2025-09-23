from torch import no_grad
from typing import Optional, Union, Any
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_response(prompt: str, model, tokenizer, args, decode_input: bool=False) -> tuple[list[str], Optional[list[str]]]:
    encoded_input = tokenizer(prompt, return_tensors="pt")
    decoded_input = None
    if decode_input:
        decoded_input = tokenizer.decode(encoded_input["input_ids"][0])
    model_inputs = encoded_input.to("cuda")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded_output[0], decoded_input


def generate_batch_responses(prompts, model, tokenizer, args, decode_inputs=False):
    encoded_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to("cuda")

    decoded_inputs = None
    if decode_inputs:
        decoded_inputs = tokenizer.batch_decode(
            encoded_inputs["input_ids"], skip_special_tokens=False
        )

    with no_grad():
        generated_ids = model.generate(
            **encoded_inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_outputs = tokenizer.batch_decode(generated_ids)
    return decoded_outputs, decoded_inputs
