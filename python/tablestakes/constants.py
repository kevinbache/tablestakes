from pathlib import Path

WORD_ID_FILENAME = 'word_to_id.json'
WORD_COUNT_FILENAME = 'word_to_count.json'
NUM_Y_CLASSES_FILENAME = 'num_y_classes.json'

THIS_DIR = Path(__file__).resolve().parent
DOCS_DIR = (THIS_DIR / '..' / '..' / 'data' / 'docs').resolve()
