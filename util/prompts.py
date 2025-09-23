from pandas import DataFrame
import random
from . import LIMIT_NUM_FUZZY_MATCHES
from typing import Callable


def _generate_instruction_prompts(
    df: DataFrame, shots: int = 0, fuzzy: bool = True
) -> list[str, str]:
    """
    Generates prompts in "instruction" format.
    Can
    Returns a tuple of lists:
    - list of English sentences (sources): "This is a car"
    - list of few-shot prompts: "<translations examples> English: This is a car\nFrench: ",
    - list French translated sentences (references or completions <=> ideal generated text): "C'est une voiture"
    """
    if shots < 0:
        raise ValueError('Argument "shots" must be non-negative integer')

    if shots > LIMIT_NUM_FUZZY_MATCHES and fuzzy:
        ValueError(
            f"Warning: Number of shots ({shots}) exceeds limit of {LIMIT_NUM_FUZZY_MATCHES}."
        )

    prompts = []

    for i in range(len(df)):
        prompt = ""
        if shots > 0:
            if fuzzy:
                for j in range(shots):
                    prompt += f'English: {df["match"][i][j][0]}\nFrench: {df["match"][i][j][1]}\n'
            else:
                random_indexes = random.sample(range(len(df)), shots)
                for idx in random_indexes:
                    prompt += f'English: {df["en"][idx]}\nFrench: {df["fr"][idx]}\n'
        prompt += f'English: {df["en"][i]}\nFrench: '
        prompts.append(prompt)

    return df["en"].to_list(), prompts, df["fr"].to_list()


def generate_prompts_for_mistral_instruct(
    df: DataFrame, shots: int = 0, fuzzy: bool = True
):
    if shots < 0:
        raise ValueError('Argument "shots" must be non-negative integer')

    if shots > LIMIT_NUM_FUZZY_MATCHES and fuzzy:
        ValueError(
            f"Warning: Number of shots ({shots}) exceeds limit of {LIMIT_NUM_FUZZY_MATCHES}."
        )

    # <s> and </s> are special tokens for beginning of string (BOS) and end of string (EOS) while [INST] and [/INST] are regular strings.
    system_prompt = "You are a translation assistant. Translate the given English sentence into French. Use the provided translations examples as guidance for style and terminology. Only output the final French translation of the English sentence, nothing else."

    prompts = []

    for i in range(len(df)):
        prompt = f"[INST] {system_prompt}"
        if shots > 0:
            if fuzzy:
                for j in range(shots):
                    prompt += f'English: {df["match"][i][j][0]}\nFrench: {df["match"][i][j][1]}\n'
            else:
                random_indexes = random.sample(range(len(df)), shots)
                for idx in random_indexes:
                    prompt += f'English: {df["en"][idx]}\nFrench: {df["fr"][idx]}\n'
        prompt += f'English: {df["en"][i]}\nFrench: [/INST]'
        prompts.append(prompt)

    return df["en"].to_list(), prompts, df["fr"].to_list()




_generators = {
    "Mistral-7B-Instruct-v0.3": generate_prompts_for_mistral_instruct
}

def get_prompt_generator(args) -> Callable:
    model_name = getattr(args, "model_name", None)
    if model_name:
        for key, gen in _generators.items():
            if key in model_name:
                print(f"Special prompt generator '{key}' was selected.")
                return gen
    print("Instruction prompt generator is used.")
    return _generate_instruction_prompts

