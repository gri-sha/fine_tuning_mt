import yaml
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_config_path = os.path.join(_project_root, "util/dataset_config.yml")

with open(_config_path, "r") as f:
    _config = yaml.safe_load(f)

MIN_LENGTH = _config["min_sentence_length"]
MIN_FUZZY_SCORE = _config["min_fuzzy_score"]
LIMIT_NUM_FUZZY_MATCHES = _config["limit_num_fuzzy_matches"]
TARGET_PATH = _config["target_path"]

SEED = _config["seed"]
TEST_SPLIT = _config["test_split"]
VALID_SPLIT = _config["valid_split"]

from .read_data import initialize_dfs, validation_split
from .prompts import generate_instruction_prompts
from .login import login_to_hf
from .convert import str_to_bool, read_shots, read_fuzzy
from .parse import parse_arguments
