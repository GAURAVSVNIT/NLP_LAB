"""Loads the Hindi POS-tagged corpus (one `word<TAB>tag` per line, blank line between sentences)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'Hindi-POS-Data'
SPLITS = ['train', 'dev', 'test']


def read_tagged(path):
    """Returns a list of sentences, each a list of (word, tag) pairs."""
    sentences, current = [], []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        word, tag = line.split('\t')
        current.append((word, tag))
    if current:
        sentences.append(current)
    return sentences


def load_split(split):
    return read_tagged(DATA_DIR / f'hindi-{split}-pos.txt')


def load_all():
    return {split: load_split(split) for split in SPLITS}
