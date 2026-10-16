"""Reads the Hindi POS-tagged corpus.

Each line is `word<TAB>tag`; a blank line separates sentences.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'Lab-Assignment-7' / 'Lab-Assignment-7' / 'Hindi-POS-Data'
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'

START = '<s>'
END = '</s>'
UNK = '<UNK>'


def read_sentences(path):
    """Returns a list of sentences, each a list of (word, tag) pairs."""
    sentences, current = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
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


def load_splits():
    return {name: read_sentences(DATA_DIR / f'hindi-{name}-pos.txt')
            for name in ('train', 'dev', 'test')}
