from pathlib import Path

WORD_ID_FILENAME = 'word_to_id.json'
WORD_COUNT_FILENAME = 'word_to_count.json'
NUM_Y_CLASSES_FILENAME = 'num_y_classes.json'

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent.resolve()

DATA_DIR = Path.home().expanduser().resolve() / 'data' / 'tablestakes'
DOCS_DIR = DATA_DIR / 'docs'
DATASETS_DIR = DATA_DIR / 'datasets'

OUTPUT_DIR = DATA_DIR / 'outputs'

CHECKPOINT_FILE_BASENAME = 'checkpoint'

LOSS_NAME = 'loss'

Y_PREFIX = 'y_'
Y_KORV_BASE_NAME = 'korv'
Y_WHICH_KV_BASE_NAME = 'which_kv'
Y_KORV_NAME = Y_PREFIX + Y_KORV_BASE_NAME
Y_WHICH_KV_NAME = Y_PREFIX + Y_WHICH_KV_BASE_NAME
Y_BASE_NAMES = [Y_KORV_BASE_NAME, Y_WHICH_KV_BASE_NAME]

X_PREFIX = 'x_'
X_BASE_BASE_NAME = 'base'
X_VOCAB_BASE_NAME = 'vocab'
X_BASE_NAME = X_PREFIX + X_BASE_BASE_NAME
X_VOCAB_NAME = X_PREFIX + X_VOCAB_BASE_NAME

META_NAME = 'meta'
META_PREFIX = f'{META_NAME}_'
META_SHORT_BASE_NAME = f'short'
META_ORIGINAL_DATA_DIR_COL_NAME = 'original_data_dir'


LOGGER_API_KEY_FILE = str(Path.home() / '.logger_api_key')
NEPTUNE_USERNAME = 'kevinbache'

SOURCES_GLOB_STR = str(THIS_DIR / '**/*.py')

# prefix assigned to parent class names when they're set on each word.
# currenttly, this class is being used to propagate the id of the kv that this word came from so name it for that.
TAG_PREFIX = 'kv_is_'


Y_VALUE_TO_IGNORE = -100


TRAIN_PHASE_NAME = 'train'
VALID_PHASE_NAME = 'valid'
TEST_PHASE_NAME = 'test'

PHASE_NAMES = [
    TRAIN_PHASE_NAME,
    VALID_PHASE_NAME,
    TEST_PHASE_NAME,
]


class ColNames:
    """DataFrame column names for input data matrices"""""

    LEFT = 'left'
    RIGHT = 'right'
    TOP = 'top'
    BOTTOM = 'bottom'
    CONFIDENCE = 'confidence'

    BBOX_NAMES = [LEFT, RIGHT, TOP, BOTTOM]

    PAGE_NUM = 'page_num'
    PAGE_WIDTH = 'page_width'
    PAGE_HEIGHT = 'page_height'
    NUM_PAGES = 'num_pages'

    # the row index value from the raw ocr csv
    RAW_OCR_INDEX = 'raw_ocr_index'

    TEXT = 'text'
    TOKEN_RAW = 'token_raw'
    TOKEN = 'token'
    TOKEN_ID = 'token_id'
    TOKENIZER = 'tok'
    # ocr tokenizes to "text", then we further tokenize into "tokens".
    # this tells us which tokens correspond to which original text items
    ORIGINAL_TEXT_INDEX = 'original_text_index'
    # WORD_WAS_ELIMINATED = 'was_removed_rare_token'

    # CharCounter
    CHAR_COUNT_PREFIX = 'num_chars_'
    LOWER_COUNT = f'{CHAR_COUNT_PREFIX}lower'
    UPPER_COUNT = f'{CHAR_COUNT_PREFIX}upper'
    NUMBER_COUNT = f'{CHAR_COUNT_PREFIX}number'
    OTHER_COUNT = f'{CHAR_COUNT_PREFIX}other'
    TOTAL_COUNT = f'{CHAR_COUNT_PREFIX}total'
    UNICODE_NORM_CHANGED_COUNT = f'{CHAR_COUNT_PREFIX}unicodechanged'
    NONASCII_COUNT = f'{CHAR_COUNT_PREFIX}nonascii'

    HAS_LEADING_ZERO = f'has_leading_zero'

    CHAR_COUNT_COLS = [
        LOWER_COUNT,
        UPPER_COUNT,
        NUMBER_COUNT,
        OTHER_COUNT,
        TOTAL_COUNT,
        UNICODE_NORM_CHANGED_COUNT,
        NONASCII_COUNT,
        HAS_LEADING_ZERO,
    ]

