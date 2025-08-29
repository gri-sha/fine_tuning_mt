import os
import sys
import yaml
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from pprint import pprint
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util import create_test_dataset, parse_translation_arguments, clean_response

args = parse_translation_arguments()

print("Translation configuration:")
pprint(args.__dict__)

with open(
    os.path.join(args.checkpoint_path, "tokenizer_config.json"), "r"
) as f:
    tok_config = json.load(f)

cache_dir = os.path.expanduser("~/.cache/huggingface/")

dataset = create_test_dataset(shots=args.shots, fuzzy=args.fuzzy)

peftconfig = PeftConfig.from_pretrained(args.checkpoint_path)

if peftconfig.task_type == "SEQ_2_SEQ_LM":
    model_base = AutoModelForSeq2SeqLM.from_pretrained(
        peftconfig.base_model_name_or_path, device_map="auto", cache_dir=cache_dir
    )
elif peftconfig.task_type == "CAUSAL_LM":
    model_base = AutoModelForCausalLM.from_pretrained(
        peftconfig.base_model_name_or_path, device_map="auto", cache_dir=cache_dir
    )
else:
    raise ValueError(f"Unknown model task: {peftconfig.task_type}")

tokenizer = AutoTokenizer.from_pretrained(
    peftconfig.base_model_name_or_path,
    cache_dir=cache_dir,
    legacy=False,
    use_fast=args.use_fast_tokenizer,
)
if args.add_bos_token is not None:
    tokenizer.add_bos_token = args.add_bos_token
if args.bos_token is not None:
    tokenizer.bos_token = args.bos_token
if args.add_eos_token is not None:
    tokenizer.add_eos_token = args.add_eos_token
if args.eos_token is not None:
    tokenizer.eos_token = args.eos_token
if args.pad_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
if args.pad_side is not None:
    tokenizer.padding_side = args.pad_side

model = PeftModel.from_pretrained(model_base, args.checkpoint_path)
print("Peft model loaded")


def generate_response(prompt, model, decode_input=False):
    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=not args.remove_special_tokens_at_encoding,
    )
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

    decoded_output = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=args.remove_special_tokens_at_decoding,
    )
    return decoded_output[0], decoded_input


def generate_batch_responses(prompts, model, decode_inputs=False):
    encoded_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=not args.remove_special_tokens_at_encoding,
    ).to("cuda")

    decoded_inputs = None
    if decode_inputs:
        decoded_inputs = tokenizer.batch_decode(
            encoded_inputs["input_ids"], skip_special_tokens=False
        )

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded_inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_outputs = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=args.remove_special_tokens_at_decoding,
    )
    return decoded_outputs, decoded_inputs


if args.do_tokenizer_check:

    print(tokenizer)
    print("Special tokens: ", tokenizer.all_special_tokens, f"({type(tokenizer.all_special_tokens)})")
    print(tokenizer.special_tokens_map, f"({type(tokenizer.special_tokens_map)})")

    # Check by sentence generation
    sentence_nums = random.sample(range(len(dataset)), 4)
    for prompt, reference in [
        (dataset["prompts"][idx], dataset["references"][idx]) for idx in sentence_nums
    ]:
        response, decoded_input = generate_response(prompt, model, decode_input=True)
        print("Decoded input: ", repr(decoded_input))
        print("Decoded output:", repr(response))
        print(
            "Cleaned output:",
            repr(
                clean_response(
                    response=response, prompt=prompt, special_tokens=tokenizer.all_special_tokens
                )
            ),
        )
        print("Reference:     ", repr(reference), "\n")
    print()

    # Check by batch generation
    idx = random.choice(range(len(dataset) - args.batch_size))
    batch = dataset["prompts"][idx : idx + args.batch_size]
    responses, decoded_inputs = generate_batch_responses(
        batch, model, decode_inputs=True
    )
    print("\nBatch responses:")
    for i in range(min(args.batch_size, 4)):
        print("Decoded input: ", repr(decoded_inputs[i]))
        print("Decoded output:", repr(responses[i]))
        print("Cleaned output:", repr(clean_response(response=responses[i], prompt=batch[i], special_tokens=tokenizer.all_special_tokens)))
        print("Reference:     ", repr(dataset["references"][idx + i]), "\n")

if args.do_translation:
    translations = []
    if args.use_batches:
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset["prompts"][i : i + args.batch_size]
            responses, _ = generate_batch_responses(batch, model, decode_inputs=False)
            for i in range(len(batch)):
                cleaned = clean_response(
                    response=responses[i], prompt=batch[i], special_tokens=tokenizer.all_special_tokens
                )
                translations.append(cleaned)
    else:
        for i in tqdm(range(len(dataset))):
            prompt = dataset["prompts"][i]
            response, _ = generate_response(prompt, model, decode_input=False)
            cleaned = clean_response(
                response=response, prompt=prompt,  special_tokens=tokenizer.all_special_tokens
            )
            translations.append(cleaned)

    directory, _ = os.path.split(args.translations_path)
    os.makedirs(directory, exist_ok=True)

    translations_df = pd.DataFrame(
        {
            "sources": dataset["sources"],
            "references": dataset["references"],
            "translations": translations,
        }
    )
    translations_df.to_csv(args.translations_path, index=False)
