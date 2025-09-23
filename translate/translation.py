import os
import sys
import json
import random
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from pprint import pprint
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util import (
    create_test_dataset,
    parse_translation_arguments,
    clean_response,
    generate_batch_responses,
    generate_response,
    load_model,
    load_tokenizer,
    login_to_hf,
    get_peft_config,
    get_quant_config,
    check_cuda_availability,
    get_prompt_generator,
)

args = parse_translation_arguments()
print("Translation configuration:")
pprint(args.__dict__)

cache_dir = os.path.expanduser("~/.cache/huggingface/")
login_to_hf()
check_cuda_availability()


dataset = create_test_dataset(
    shots=args.shots, fuzzy=args.fuzzy, prompt_generator=get_prompt_generator(args)
)
pprint(dataset)

if args.use_base_model:
    quant_config = get_quant_config(args)
    pprint(quant_config)

    peft_config = get_peft_config(args)
    pprint(peft_config)

    model = load_model(args, quant_config, peft_config, cache_dir, inference=False)
    tokenizer = load_tokenizer(args, cache_dir)
else:
    with open(os.path.join(args.checkpoint_path, "tokenizer_config.json"), "r") as f:
        tok_config = json.load(f)

    peft_config = PeftConfig.from_pretrained(args.checkpoint_path)

    if peft_config.task_type == "SEQ_2_SEQ_LM":
        model_base = AutoModelForSeq2SeqLM.from_pretrained(
            peft_config.base_model_name_or_path, device_map="auto", cache_dir=cache_dir
        )
    elif peft_config.task_type == "CAUSAL_LM":
        model_base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, device_map="auto", cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unknown model task: {peft_config.task_type}")

    tokenizer = load_tokenizer(
        args, cache_dir, model_name=peft_config.base_model_name_or_path
    )

    model = PeftModel.from_pretrained(model_base, args.checkpoint_path)
    print("Model loaded")

if args.do_inference_check:
    print(tokenizer.__class__.__name__)
    print(
        "Special tokens: ",
        tokenizer.all_special_tokens,
        f"({type(tokenizer.all_special_tokens)})",
    )
    print(tokenizer.special_tokens_map, f"({type(tokenizer.special_tokens_map)})")

    # Check by sentence generation
    sentence_nums = random.sample(range(len(dataset)), 4)
    for prompt, reference in [
        (dataset["prompts"][idx], dataset["references"][idx]) for idx in sentence_nums
    ]:
        response, decoded_input = generate_response(
            prompt, model, tokenizer, args, decode_input=True
        )
        print("Decoded input: ", repr(decoded_input))
        print("Decoded output:", repr(response))
        print(
            "Cleaned output:",
            repr(
                clean_response(
                    response=response,
                    prompt=prompt,
                    special_tokens=tokenizer.all_special_tokens,
                )
            ),
        )
        print("Reference:     ", repr(reference), "\n")
    print()

    # Check by batch generation
    idx = random.choice(range(len(dataset) - args.batch_size))
    batch = dataset["prompts"][idx : idx + args.batch_size]
    responses, decoded_inputs = generate_batch_responses(
        batch, model, tokenizer, args, decode_inputs=True
    )
    print("\nBatch responses:")
    for i in range(min(args.batch_size, 4)):
        print("Decoded input: ", repr(decoded_inputs[i]))
        print("Decoded output:", repr(responses[i]))
        print(
            "Cleaned output:",
            repr(
                clean_response(
                    response=responses[i],
                    prompt=batch[i],
                    special_tokens=tokenizer.all_special_tokens,
                )
            ),
        )
        print("Reference:     ", repr(dataset["references"][idx + i]), "\n")

if args.do_translation:
    translations = []
    if args.use_batches:
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset["prompts"][i : i + args.batch_size]
            responses, _ = generate_batch_responses(
                batch, model, tokenizer, args, decode_inputs=False
            )
            for i in range(len(batch)):
                cleaned = clean_response(
                    response=responses[i],
                    prompt=batch[i],
                    special_tokens=tokenizer.all_special_tokens,
                )
                translations.append(cleaned)
    else:
        for i in tqdm(range(len(dataset))):
            prompt = dataset["prompts"][i]
            response, _ = generate_response(
                prompt, model, tokenizer, args, decode_input=False
            )
            cleaned = clean_response(
                response=response,
                prompt=prompt,
                special_tokens=tokenizer.all_special_tokens,
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
    print(f"Translations saved to {args.translations_path}")
