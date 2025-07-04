import os
import pandas as pd
from util import MIN_LENGTH, TARGET_PATH
from .fuzzy_matches import calculate_fuzzy_matches


def _create_df() -> pd.DataFrame:
    total_written = 0
    total_read = 0

    df = pd.DataFrame([], columns=["en", "fr"])

    for i in range(1, 5):
        en = open(f"data/en/book{i}_en.txt", "r")
        fr = open(f"data/fr/book{i}_fr.txt", "r")

        book_lines_read = 0

        while True:
            en_line = en.readline()
            fr_line = fr.readline()

            if not en_line or not fr_line:
                break

            book_lines_read += 1

            en_line = en_line.strip()
            fr_line = fr_line.strip()

            if len(en_line) < MIN_LENGTH or len(fr_line) < MIN_LENGTH:
                continue

            df = pd.concat(
                [pd.DataFrame([[en_line, fr_line]], columns=df.columns), df],
                ignore_index=True,
            )
            total_written += 1

        total_read += book_lines_read
        print(f"Book {i}: {book_lines_read} lines read.")

    print()
    print(f"Total lines read: {total_read}")
    print(f"Total lines written: {total_written}")

    return df


def initialize_dataset() -> pd.DataFrame:
    if os.path.exists(TARGET_PATH):
        print("Constructed dataset found. Reading source...")
        df = pd.read_csv(TARGET_PATH)
        if not df.empty:
            print("Non-empty dataset read.")
            return df
        else:
            print("Empty dataset is read, reconstructing...")
    else:
        print("No constructed dataset found. Reading sources...")

    df = _create_df()
    print("Calculating fuzzy-matches...")
    df = calculate_fuzzy_matches(df)
    df.to_csv(TARGET_PATH, index=False)
    return df
