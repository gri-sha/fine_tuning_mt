import os
import pandas as pd
from datasets import Dataset, DatasetDict
from util import (
    _project_root,
    MIN_LENGTH,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
)
from .fuzzy_matches import calculate_fuzzy_matches
from .prompts import _generate_instruction_prompts
from typing import Optional


def _split(df, train, validation, test):
    if not abs((train + validation + test) - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    split1 = int(len(df) * train)
    split2 = int(len(df) * (train + validation))

    df_train = df[:split1]
    df_val = df[split1:split2].reset_index(drop=True)
    df_test = df[split2:].reset_index(drop=True)

    return df_train, df_val, df_test


def _create_dfs() -> None:
    print("\nCreating dataframes...")
    total_written = 0
    total_read = 0
    df = pd.DataFrame([], columns=["en", "fr"])

    en = open(os.path.join(_project_root, "data/all_en.txt"), "r")
    fr = open(os.path.join(_project_root, "data/all_fr.txt"), "r")

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

    df_train, df_val, df_test = _split(df, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT)

    print(f"\nTotal lines read: {total_read}")
    print(f"Train lines written: {len(df_train)} to {TRAIN_PATH}")
    print(f"Validation lines written: {len(df_val)} to {VALIDATION_PATH}")
    print(f"Test lines written: {len(df_test)} to {TEST_PATH}")
    print(f"Total lines written: {total_written}")

    print("\nCalculating fuzzy matches...")
    # for training we use only mathches from training dataset (to prevznt training on validation andn test data)
    df_train = calculate_fuzzy_matches(df=df_train, df_to_choose_from=df_train)
    # for validation and test we can use matches from the whole dataset
    df_val = calculate_fuzzy_matches(df=df_val, df_to_choose_from=df)
    df_test = calculate_fuzzy_matches(df=df_test, df_to_choose_from=df)
    print("Fuzzy matches calculated.")

    print("\nSaving dataframes with pickle...")
    df_train.to_pickle(os.path.join(_project_root, TRAIN_PATH))
    df_val.to_pickle(os.path.join(_project_root, VALIDATION_PATH))
    df_test.to_pickle(os.path.join(_project_root, TEST_PATH))


def _initialize_dfs(recreate=False) -> tuple[pd.DataFrame]:
    train_path = os.path.join(_project_root, TRAIN_PATH)
    val_path = os.path.join(_project_root, VALIDATION_PATH)
    test_path = os.path.join(_project_root, TEST_PATH)

    df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_exist = all(os.path.exists(p) for p in [train_path, val_path, test_path])

    if not all_exist or recreate:

        if recreate:
            print("\nRecreating dataframes...")
        else:
            print("\nOne or more dataframes missing. Recreating all...")

        for p in [train_path, val_path, test_path]:
            if os.path.exists(p):
                os.remove(p)

        _create_dfs()
    else:
        print("\nLoading existing dataframes...")

    df_train = pd.read_pickle(train_path)
    df_val = pd.read_pickle(val_path)
    df_test = pd.read_pickle(test_path)
    print("Dataframes loaded.")
    
    if df_train.empty or df_val.empty or df_test.empty:
        print("Warning: One of the dataframes is empty.")
    return df_train, df_val, df_test


def _build_prompt_completion_dataset(
    df_part, shots: list[int], fuzzy: list[Optional[bool]]
) -> Dataset:
    prompts = []
    completions = []
    for s, f in zip(shots, fuzzy):
        if f is None and s != 0:
            raise ValueError("If 'fuzzy' is None, 'shots' must be 0.")
        _, p, c = _generate_instruction_prompts(df_part, shots=s, fuzzy=f)
        prompts.extend(p)
        completions.extend(c)
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def create_train_dataset(
    shots: list[int],
    fuzzy: list[Optional[bool]],
) -> DatasetDict:
    df_train, df_val, _ = _initialize_dfs()
    train_dataset = _build_prompt_completion_dataset(df_train, shots, fuzzy)
    val_dataset = _build_prompt_completion_dataset(df_val, shots, fuzzy)
    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def create_test_dataset(shots: list[int], fuzzy: list[Optional[bool]]) -> Dataset:
    _, _, df_test = _initialize_dfs()
    sources = []
    references = []
    prompts = []

    if len(shots) != len(fuzzy):
        raise ValueError(
            "The number of 'shots' must match the number of 'fuzzy' values."
        )

    for s, f in zip(shots, fuzzy):
        if f is None and s != 0:
            raise ValueError("If 'fuzzy' is None, 'shots' must be 0.")
        s, p, r = _generate_instruction_prompts(df_test, shots=s, fuzzy=f)
        sources.extend(s)
        references.extend(r)
        prompts.extend(p)

    return Dataset.from_dict(
        {"sources": sources, "references": references, "prompts": prompts}
    )
