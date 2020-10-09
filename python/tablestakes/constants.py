from pathlib import Path

WORD_ID_FILENAME = 'word_to_id.json'
WORD_COUNT_FILENAME = 'word_to_count.json'
NUM_Y_CLASSES_FILENAME = 'num_y_classes.json'

THIS_DIR = Path(__file__).resolve().parent
DOCS_DIR = (THIS_DIR / '..' / '..' / 'data' / 'docs').resolve()
Y_KORV_NAME = 'y_korv'
Y_WHICH_KV_NAME = 'y_which_kv'
Y_PREFIX = 'y_'

Y_BASE_NAMES = [Y_KORV_NAME.replace(Y_PREFIX, ''), Y_WHICH_KV_NAME.replace(Y_PREFIX, '')]

X_PREFIX = 'x_'
X_BASIC_BASE_NAME = 'basic'
X_VOCAB_BASE_NAME = 'vocab'
X_BASIC_NAME = X_PREFIX + X_BASIC_BASE_NAME
X_VOCAB_NAME = X_PREFIX + X_VOCAB_BASE_NAME

LOGS_DIR = THIS_DIR.parent.parent.resolve() / 'logs/'

CHECKPOINT_FILE_BASENAME = 'checkpoint'