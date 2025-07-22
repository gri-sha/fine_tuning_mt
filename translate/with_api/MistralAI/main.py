import os
import sys
import yaml
import pandas as pd
from mistralai import Mistral
from pprint import pprint
from dotenv import load_dotenv

from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from util import generate_instruction_prompts, initialize_dfs, TEST_SPLIT

with open("translate/with_api/MistralAI/mistralAI_config.yml", "r") as f:
    config = yaml.safe_load(f)

_, df_test = initialize_dfs(test=TEST_SPLIT)
s0, p0, r0 = generate_instruction_prompts(df_test, shots=0)
s1, p1, r1 = generate_instruction_prompts(df_test, shots=1, fuzzy=True)

dataset_0shot = pd.DataFrame({"sources": s0, "references": r0, "prompts": p0})
dataset_1shot = pd.DataFrame({"sources": s1, "references": r1, "prompts": p1})

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

for row in dataset_0shot.head(3).itertuples():
    print(row.prompts)

    chat_response = client.chat.complete(
        model= config['model'],
        temperature=config['temperature'],
        max_tokens=config['max_tokens'],
        messages = [
            {
                "role": "system",
                "content": config['system_prompt'],
            },
            {
                "role": "user",
                "content": row.prompts,
            },
        ]
    )

    print(chat_response)
    print(chat_response.choices[0].message.content)

