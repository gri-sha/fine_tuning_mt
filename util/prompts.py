from pandas import DataFrame
import random


def generate_training_prompts(
    df: DataFrame, shots: int = 0, fuzzy: bool = True
) -> list[str]:
    """
    Returns a list of constructed prompt.
    """
    if shots < 0:
        raise ValueError('Argument "shots" must be non-negative integer')

    prompts = []
    for i in range(len(df)):
        prompt = f'English: {df["en"][i]}\nFrench: {df["fr"][i]}\n'
        if shots > 0:
            if fuzzy:
                for j in range(shots):
                    prompt += f'English: {df["match"][i][j][0]}\nFrench: {df["match"][i][j][1]}\n'
            else:
                random_indexes = random.sample(range(len(df)), shots)
                for idx in random_indexes:
                    prompt += f'English: {df["en"][idx]}\nFrench: {df["fr"][idx]}\n'
        prompts.append(prompt)
    return prompts


def generate_eval_prompts(
    df: DataFrame, shots: int = 0, fuzzy: bool = True
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns a tuple of lists:
    - list of English sentences (sources),
    - list of French translated sentneces (references),
    - list of constructed prompts,
    """
    if shots < 0:
        raise ValueError('Argument "shots" must be non-negative integer')

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
    return df["en"].to_list(), df["fr"].to_list(), prompts


# def generate_instructive_prompt(target: str, examples: list[tuple[str]] = []) -> str:
#     # Pretrained, but not fine tuned model
#     # Tags [INST], <s> are used only for fine tuned Mistral 7B (e.g. Mistral-7B-Instruct)
#     if examples:
#         header = "You are a helpful translation assistant. Your task is to translate the given English sentence to French based on the following examples:\n"
#         shots = ""
#         for elem in examples:
#             shots += f"English: {elem[0]} -> French: {elem[1]}\n"
#         footer = f"Output only the translated sentence without explanations.\nTranslate the following sentence to French:\n{target}"
#         return header + shots + footer
#     else:
#         return f"You are a helpful translation assistant. Your task is to translate the given English sentence to French.\nOutput only the translated sentence without explanations.\nTranslate the following sentence to French:\n{target}"
