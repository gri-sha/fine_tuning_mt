import pandas as pd
from thefuzz import fuzz, process
from util import MIN_FUZZY_SCORE, LIMIT_NUM_FUZZY_MATCHES


def get_fuzzy_matches(phrase, choices_df, limit, min_score):
    # Filter out the exact same phrase
    filtered_df = choices_df[choices_df["en"] != phrase].copy()

    # Build lookup dict for English → French
    en_to_fr = dict(zip(filtered_df["en"], filtered_df["fr"]))

    # Get fuzzy matches against English phrases
    matches = process.extract(
        phrase, en_to_fr.keys(), limit=limit, scorer=fuzz.ratio
    )

    # Filter by minimum score and format result
    result = [
        (match_phrase, en_to_fr[match_phrase], score)
        for match_phrase, score in matches
        if score >= min_score
    ]

    return result


def calculate_fuzzy_matches(
    df: pd.DataFrame, df_to_choose_from: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, "match"] = df["en"].apply(
        lambda x: get_fuzzy_matches(
            x,
            df_to_choose_from,
            limit=LIMIT_NUM_FUZZY_MATCHES,
            min_score=MIN_FUZZY_SCORE,
        )
    )
    return df

