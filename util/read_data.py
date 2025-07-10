import os
import ast
import pandas as pd
from datasets import Dataset, DatasetDict
from util import MIN_LENGTH, TARGET_PATH
from .fuzzy_matches import calculate_fuzzy_matches


def _create_df() -> pd.DataFrame:
    total_written = 0
    total_read = 0

    df = pd.DataFrame([], columns=["en", "fr"])

    en = open(f"data/all_en.txt", "r")
    fr = open(f"data/all_fr.txt", "r")

    while True:
        en_line = en.readline()
        fr_line = fr.readline()

        if not en_line or not fr_line:
            break

        total_read += 1

        en_line = en_line.strip()
        fr_line = fr_line.strip()

        if len(en_line) < MIN_LENGTH or len(fr_line) < MIN_LENGTH:
            continue

        df = pd.concat(
            [pd.DataFrame([[en_line, fr_line]], columns=df.columns), df],
            ignore_index=True,
        )
        total_written += 1

    print()
    print(f"Total lines read: {total_read}")
    print(f"Total lines written: {total_written}")

    return df


def initialize_dfs(test: float) -> pd.DataFrame:
    if os.path.exists(TARGET_PATH):
        print("Loading existing dataframe...")
        df = pd.read_csv(TARGET_PATH)
        df["match"] = df["match"].apply(ast.literal_eval)
        if not df.empty:
            print("Dataframe loaded.")
            split = int(len(df) * (1 - test))
            df_train, df_test = df[:split], df[split:]
            df_test = df_test.reset_index(drop=True)
            print(f"Split at index {split}.")
            return (df_train, df_test)
        else:
            print("Dataframe is empty. Rebuilding...")
    else:
        print("No existing dataframe. Creating new one...")

    df = _create_df()
    print("Running fuzzy matching...")
    df = calculate_fuzzy_matches(df)
    df.to_csv(TARGET_PATH, index=False)
    print("Done.")
    split = int(len(df) * (1 - test))
    df_train, df_test = df[:split], df[split:]
    df_test = df_test.reset_index(drop=True)
    print(f"Split at index {split}.")
    return (df_train, df_test)


def validation_split(dataset: Dataset, validation: float) -> DatasetDict:
    split = dataset.train_test_split(test_size=validation, shuffle=True)
    return DatasetDict({"train": split["train"], "validation": split["test"]})
