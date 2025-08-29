from typing import Optional

def clean_response(response: str, prompt: Optional[str] = None, special_tokens: Optional[list[str]] = None, other_symbols: Optional[list[str]] = None) -> str:
    if prompt is not None:
        response = response.replace(prompt, "")
    if special_tokens is not None:
        for token in special_tokens:
            response = response.replace(token, "")
    if other_symbols is not None:
        for symbol in other_symbols:
            response = response.replace(symbol, "")
    return response.strip()

def get_list_of_special_tokens(tokenizer) -> list[str]:
    pass