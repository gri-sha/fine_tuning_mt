import yaml
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_config_path = os.path.join(_project_root, "util/dataset_config.yml")

with open(_config_path, "r") as f:
    dataset_config = yaml.safe_load(f)

MIN_LENGTH = dataset_config["min_sentence_length"]
MIN_FUZZY_SCORE = dataset_config["min_fuzzy_score"]
LIMIT_NUM_FUZZY_MATCHES = dataset_config["limit_num_fuzzy_matches"]

TRAIN_PATH = dataset_config["train_data_path"]
VALIDATION_PATH = dataset_config["validation_data_path"]
TEST_PATH = dataset_config["test_data_path"]

TRAIN_SPLIT = dataset_config["train_split"]
TEST_SPLIT = dataset_config["test_split"]
VALID_SPLIT = dataset_config["valid_split"]

from .read_data import (
    create_train_dataset,
    create_test_dataset,
    _initialize_dfs,
)
from .login import login_to_hf
from .parse import parse_training_arguments, parse_translation_arguments
from .prompts import _generate_instruction_prompts
from .clean import clean_response
from .metrics import plot_log_metrics
