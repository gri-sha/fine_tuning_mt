import json
import os

_config_path = "dataset_config.json"

with open(_config_path, "r") as f:
    _config = json.load(f)

MIN_LENGTH = _config["min_sentence_length"]
MIN_FUZZY_SCORE = _config["min_fuzzy_score"]
LIMIT_NUM_FUZZY_MATCHES = _config["limit_num_fuzzy_matches"]
TARGET_PATH = _config["target_path"]

from .read_data import initialize_dfs, validation_split
from .prompts import generate_training_prompts, generate_eval_prompts
