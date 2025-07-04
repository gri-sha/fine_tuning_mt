import json
import os

_current_dir = os.path.dirname(__file__)
_config_path = os.path.join(_current_dir, "config.json")

with open(_config_path, "r") as f:
    _config = json.load(f)

MIN_LENGTH = _config["min_length"]
MIN_FUZZY_SCORE = _config["min_fuzzy_score"]
LIMIT_NUM_FUZZY_MATCHES = _config["limit_num_fuzzy_matches"]
TARGET_PATH = _config["target_path"]


from util.create_dataset import initialize_dataset

__all__ = ["initialize_dataset"]
