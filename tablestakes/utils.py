from pathlib import Path
from typing import *


def save_txt(contents: Text, filename: Union[Text, Path]):
    with open(filename, 'w') as f:
        f.write(contents)
