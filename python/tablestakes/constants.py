from pathlib import Path

from tablestakes import utils

WORD_ID_FILENAME = 'word_to_id.json'
WORD_COUNT_FILENAME = 'word_to_count.json'
NUM_Y_CLASSES_FILENAME = 'num_y_classes.json'

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent.resolve()

DATA_DIR = Path.home().expanduser().resolve() / 'data' / 'tablestakes'
DOCS_DIR = DATA_DIR / 'docs'
DATASETS_DIR = DATA_DIR / 'datasets'

LOGS_DIR = DATA_DIR / 'logs'

CHECKPOINT_FILE_BASENAME = 'checkpoint'

Y_PREFIX = 'y_'
Y_KORV_BASE_NAME = 'korv'
Y_WHICH_KV_BASE_NAME = 'which_kv'
Y_KORV_NAME = Y_PREFIX + Y_KORV_BASE_NAME
Y_WHICH_KV_NAME = Y_PREFIX + Y_WHICH_KV_BASE_NAME

Y_BASE_NAMES = [Y_KORV_BASE_NAME, Y_WHICH_KV_BASE_NAME]

X_PREFIX = 'x_'
X_BASIC_BASE_NAME = 'basic'
X_VOCAB_BASE_NAME = 'vocab'
X_BASIC_NAME = X_PREFIX + X_BASIC_BASE_NAME
X_VOCAB_NAME = X_PREFIX + X_VOCAB_BASE_NAME

META_DIR_NAME = 'meta'

LOGGER_API_KEY_FILE = str(Path.home() / '.logger_api_key')
NEPTUNE_USERNAME = 'kevinbache'

SOURCES_GLOB_STR = str(THIS_DIR / '**/*.py')
