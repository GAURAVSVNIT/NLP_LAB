"""Sample sentences from Assignment-1's tokenized Gujarati corpus and split
them into train/dev/test sets for the Lab-4 n-gram language models.

Source: Assignment_1/output_full/tokenized_sentences.parquet, one already
word-tokenized sentence per row (23,594,135 rows total). We take a
systematic sample spread evenly across the whole file (rather than just the
first N rows) so the sample isn't biased toward whatever document happened
to be processed first, then shuffle with a fixed seed and slice off dev/test.
"""
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PARQUET = ROOT.parent / 'Assignment_1' / 'output_full' / 'tokenized_sentences.parquet'
OUTPUT_DIR = ROOT / 'output'

N_TOTAL = 1_000_000
N_DEV = 1_000
N_TEST = 1_000
SEED = 42


def systematic_sample(path: Path, n_total: int) -> list[str]:
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    stride = max(1, n_rows // n_total)

    sentences = []
    row_idx = 0
    for batch in pf.iter_batches(batch_size=200_000, columns=['sentence']):
        col = batch.column('sentence')
        batch_len = len(col)
        # first offset in this batch that lands on the stride grid
        first = (-row_idx) % stride
        for i in range(first, batch_len, stride):
            val = col[i].as_py()
            if val:
                sentences.append(val)
        row_idx += batch_len
        if len(sentences) >= n_total:
            break
    return sentences[:n_total]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f'Sampling {N_TOTAL:,} sentences from {SOURCE_PARQUET} ...')
    sentences = systematic_sample(SOURCE_PARQUET, N_TOTAL)
    print(f'Got {len(sentences):,} sentences.')

    rng = random.Random(SEED)
    rng.shuffle(sentences)

    dev = sentences[:N_DEV]
    test = sentences[N_DEV:N_DEV + N_TEST]
    train = sentences[N_DEV + N_TEST:]

    for name, split in [('train', train), ('dev', dev), ('test', test)]:
        out_path = OUTPUT_DIR / f'{name}.txt'
        out_path.write_text('\n'.join(split) + '\n', encoding='utf-8')
        print(f'{name}: {len(split):,} sentences -> {out_path}')

    meta = {
        'source': str(SOURCE_PARQUET),
        'n_total': len(sentences),
        'n_train': len(train),
        'n_dev': len(dev),
        'n_test': len(test),
        'seed': SEED,
    }
    (OUTPUT_DIR / 'split_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
