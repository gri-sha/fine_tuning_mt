from typing import Optional

def str_to_bool(value: str) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    elif value == '_':
        return None
    else:
        raise ValueError(f"Boolean value expected, got: {value}")
    
def read_shots(shots: str) -> list[int]:
    # Reads cli argument "0 1 3" as [0, 1, 3]
    return list(map(int, shots.strip().split()))

def read_fuzzy(fuzzy: str) -> list[Optional[bool]]:
    if fuzzy == "":
        return [None]
    return [str_to_bool(x) for x in fuzzy.split()]
