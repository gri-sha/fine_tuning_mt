import pandas as pd
from thefuzz import fuzz, process
from util import MIN_FUZZY_SCORE, LIMIT_NUM_FUZZY_MATCHES


def _get_fuzzy_matches(
    phrase, choices, limit=LIMIT_NUM_FUZZY_MATCHES, min_score=MIN_FUZZY_SCORE
):
    filtered_choices = [choice for choice in choices if choice != phrase]
    matches = process.extract(phrase, filtered_choices, limit=limit, scorer=fuzz.ratio)
    matches = [match for match in matches if match[1] >= min_score]
    return matches


def calculate_fuzzy_matches(df):
    df["match"] = df["en"].apply(
        lambda x: _get_fuzzy_matches(
            x,
            df["en"].tolist(),
            limit=LIMIT_NUM_FUZZY_MATCHES,
            min_score=MIN_FUZZY_SCORE,
        )
    )
    return df
