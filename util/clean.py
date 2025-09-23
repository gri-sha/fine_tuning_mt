from typing import Optional


def clean_response(
    response: str,
    prompt: Optional[str] = None,
    special_tokens: Optional[list[str]] = None,
    other_symbols: Optional[list[str]] = None,
) -> str:
    if prompt is not None:
        # normalize whitespace for reliable matching
        response_norm = " ".join(response.split())
        prompt_norm = " ".join(prompt.split())
        response = response_norm.replace(prompt_norm, "")
    if special_tokens:
        for token in special_tokens:
            response = response.replace(token, "")
    if other_symbols:
        for symbol in other_symbols:
            response = response.replace(symbol, "")

    return response.strip()
