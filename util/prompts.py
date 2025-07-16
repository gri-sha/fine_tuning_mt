from pandas import DataFrame
import random

def generate_instruction_prompts(
    df: DataFrame, shots: int = 0, fuzzy: bool = True
) -> list[str, str]:
    """
    Generates prompts in "instruction" format.
    Can
    Returns a tuple of lists:
    - list of English sentences (sources): "This is a car" 
    - list of few-shot prompts: "<translations examples> English: This is a car\nFrench: ",
    - list French translated sentences (references or completions <=> "ideal generated text"): "C'est une voiture"
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

    return df["en"].to_list(), prompts, df["fr"].to_list()
