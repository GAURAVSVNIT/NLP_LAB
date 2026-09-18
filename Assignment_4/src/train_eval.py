"""Train unigram/bigram/trigram/quadrigram models (Laplace smoothing) on
output/train.txt and evaluate perplexity on output/dev.txt and output/test.txt.
"""
import json
import time
from pathlib import Path

from ngram_model import NgramLanguageModel, read_sentences

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ORDER_NAMES = {1: 'unigram', 2: 'bigram', 3: 'trigram', 4: 'quadrigram'}


def main():
    t0 = time.time()
    train = read_sentences(OUTPUT_DIR / 'train.txt')
    dev = read_sentences(OUTPUT_DIR / 'dev.txt')
    test = read_sentences(OUTPUT_DIR / 'test.txt')
    print(f'train={len(train):,}  dev={len(dev):,}  test={len(test):,} sentences')

    model = NgramLanguageModel(orders=(1, 2, 3, 4))
    freq = model.build_vocab(train, min_count=2)
    print(f'Vocabulary (kept words + <UNK> + </s>): V = {model.V:,}  '
          f'(raw types in train: {len(freq):,})')

    print('Counting n-grams over the training split ...')
    model.fit(train)
    print(f'Total scored training tokens: {model.total_tokens:,}  '
          f'({time.time() - t0:.1f}s so far)')

    results = {
        'vocab_size': model.V,
        'raw_train_types': len(freq),
        'train_tokens': model.total_tokens,
        'n_train_sentences': len(train),
        'n_dev_sentences': len(dev),
        'n_test_sentences': len(test),
        'dev_oov_rate': model.oov_rate(dev),
        'test_oov_rate': model.oov_rate(test),
        'perplexity': {},
    }

    for n in (1, 2, 3, 4):
        dev_pp, dev_tok = model.perplexity(n, dev)
        test_pp, test_tok = model.perplexity(n, test)
        results['perplexity'][ORDER_NAMES[n]] = {
            'order': n,
            'dev_perplexity': dev_pp,
            'test_perplexity': test_pp,
            'scored_tokens': dev_tok,
            'n_unique_ngrams': len(model.ngram_counts[n]),
        }
        print(f'{ORDER_NAMES[n]:>10} (n={n}): dev PP={dev_pp:12.2f}  '
              f'test PP={test_pp:12.2f}  unique {n}-grams={len(model.ngram_counts[n]):,}')

    # A few example next-word predictions from the trigram model, for the report.
    # Build a context -> [(next_word, count), ...] index in one pass, rather than
    # rescanning all trigram counts per example.
    context_next = {}
    for ngram, count in model.ngram_counts[3].items():
        context_next.setdefault(ngram[:2], []).append((ngram[-1], count))

    examples = []
    for sent in dev[:5]:
        if len(sent) < 3:
            continue
        w1, w2 = sent[0], sent[1]
        ctx = tuple(model._map(w) for w in (w1, w2))
        candidates = sorted(context_next.get(ctx, []), key=lambda p: p[1], reverse=True)[:5]
        examples.append({'context': [w1, w2], 'top_next_words': [w for w, _ in candidates]})
    results['trigram_examples'] = examples

    out_path = OUTPUT_DIR / 'results.json'
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nWrote {out_path}  (total time {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
