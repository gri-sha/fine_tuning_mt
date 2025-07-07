import pandas as pd
from thefuzz import fuzz, process
from util import MIN_FUZZY_SCORE, LIMIT_NUM_FUZZY_MATCHES


def get_fuzzy_matches(
    phrase, choices_df, limit=LIMIT_NUM_FUZZY_MATCHES, min_score=MIN_FUZZY_SCORE
):
    # Filter out the exact same phrase
    filtered_df = choices_df[choices_df["en"] != phrase].copy()

    # Get fuzzy matches against English phrases
    matches = process.extract(
        phrase, filtered_df["en"].tolist(), limit=limit, scorer=fuzz.ratio
    )

    # Filter by minimum score
    matches = [match for match in matches if match[1] >= min_score]

    # Convert to the desired format: (english_phrase, french_phrase, score)
    result = []
    for match_phrase, score in matches:  # Skip the score
        # Find the corresponding French phrase
        french_phrase = filtered_df[filtered_df["en"] == match_phrase]["fr"].iloc[0]
        result.append((match_phrase, french_phrase))

    return result


def calculate_fuzzy_matches(df):
    df["match"] = df["en"].apply(
        lambda x: get_fuzzy_matches(
            x,
            df,  # pass the entire dataframe
            limit=LIMIT_NUM_FUZZY_MATCHES,
            min_score=MIN_FUZZY_SCORE,
        )
    )
    return df
