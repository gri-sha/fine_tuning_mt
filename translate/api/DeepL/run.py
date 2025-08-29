import os
import sys
import yaml
import pandas as pd
import deepl
from dotenv import load_dotenv
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from util import _initialize_dfs

with open("translate/with_api/DeepL/deepl_config.yml", "r") as f:
    config = yaml.safe_load(f)

_, _, df_test = _initialize_dfs()

load_dotenv()
translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))

batch_size = config["batch_size"]
translations = []

for i in range(0, len(df_test), batch_size):
    batch = df_test["en"].iloc[i : i + batch_size].tolist()
    results = translator.translate_text(
        batch,
        source_lang="EN",
        target_lang="FR",
        formality=config["formality"],
        preserve_formatting=config["preserve_formatting"],
        model_type=config["model_type"]
    )
    translations.extend([res.text for res in results])

df = pd.DataFrame({
    "sources": df_test["en"],
    "references": df_test["fr"],
    "translations": translations
})

directory, _ = os.path.split(config["translations_path"])
os.makedirs(directory, exist_ok=True)
df.to_csv(config["translations_path"], index=False)
