from pandas import DataFrame
import random


def generate_simple_prompts(dataset: DataFrame, shots: int = 0, fuzzy: bool = True) -> list[str]:
    if shots < 0:
        raise ValueError('Argument "shots" must be non-negative integer')

    prompts = []
    for i in range(len(dataset)):
        prompt = f'English: {dataset["en"][i]}\nFrench: {dataset["fr"][i]}\n'
        if shots > 0:
            if fuzzy:
                for j in range(shots):
                    prompt += f'English: {dataset["match"][i][j][0]}\nFrench: {dataset["match"][i][j][1]}\n'
            else:
                random_indexes = random.sample(range(len(dataset)), shots)
                for idx in random_indexes:
                    prompt += f'English: {dataset["en"][idx]}\nFrench: {dataset["fr"][idx]}\n'
        prompts.append(prompt)
    return prompts


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
