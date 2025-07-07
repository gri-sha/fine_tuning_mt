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


def initialize_df() -> pd.DataFrame:
    if os.path.exists(TARGET_PATH):
        print("Constructed dataframe found. Reading source...")
        df = pd.read_csv(TARGET_PATH)
        df["match"] = df["match"].apply(ast.literal_eval)
        if not df.empty:
            print("Non-empty dataframe read.")
            return df
        else:
            print("Empty dataframe is read, reconstructing...")
    else:
        print("No constructed dataframe found. Reading sources...")

    df = _create_df()
    print("Calculating fuzzy-matches...")
    df = calculate_fuzzy_matches(df)
    df.to_csv(TARGET_PATH, index=False)
    print("Dataframe created")

    return df


def split_dataset(dataset: Dataset, train, test, validation) -> DatasetDict:
    dataset = dataset.train_test_split(test_size=test, shuffle=True)
    dataset["train"], dataset["validation"] = (
        dataset["train"]
        .train_test_split(test_size=(validation / train), shuffle=True)
        .values()
    )

    return dataset
