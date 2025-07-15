from pandas import DataFrame
import random


def generate_simple_training_prompts(
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


def generate_simple_eval_prompts(
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

def generate_prompts_as_text_and_labels(df: DataFrame, shots: int = 0, fuzzy: bool = True) -> list[str, str]:
    if shots < 0:
        raise ValueError('Argument "shots" must be non-negative integer')

    prompts = []
    labels = []

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
        labels.append(prompt + df["fr"][i])
    return prompts, labels
