"""Train unigram/bigram/trigram/quadrigram models on Assignment 4's
train/dev/test split and evaluate Add-k (k=0.3) smoothed perplexity,
alongside the Assignment 4 Laplace (k=1) numbers for comparison.

Reuses Assignment_4/output/{train,dev,test}.txt directly rather than
re-sampling/re-splitting the corpus, since the assignment asks to apply
Add-k smoothing "to all 4 models you developed in the previous assignment"
-- same data, same vocabulary, only the smoothing constant changes.
"""
import json
import time
from pathlib import Path

from ngram_model import NgramLanguageModel, read_sentences

ROOT = Path(__file__).resolve().parent.parent
ASSIGNMENT_4_OUTPUT = ROOT.parent / 'Assignment_4' / 'output'
OUTPUT_DIR = ROOT / 'output'
ORDER_NAMES = {1: 'unigram', 2: 'bigram', 3: 'trigram', 4: 'quadrigram'}
K = 0.3


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    train = read_sentences(ASSIGNMENT_4_OUTPUT / 'train.txt')
    dev = read_sentences(ASSIGNMENT_4_OUTPUT / 'dev.txt')
    test = read_sentences(ASSIGNMENT_4_OUTPUT / 'test.txt')
    print(f'train={len(train):,}  dev={len(dev):,}  test={len(test):,} sentences '
          f'(reused from {ASSIGNMENT_4_OUTPUT})')

    model = NgramLanguageModel(orders=(1, 2, 3, 4))
    freq = model.build_vocab(train, min_count=2)
    print(f'Vocabulary (kept words + <UNK> + </s>): V = {model.V:,}  '
          f'(raw types in train: {len(freq):,})')

    print('Counting n-grams over the training split ...')
    model.fit(train)
    print(f'Total scored training tokens: {model.total_tokens:,}  '
          f'({time.time() - t0:.1f}s so far)')

    laplace_results = json.loads((ASSIGNMENT_4_OUTPUT / 'results.json').read_text(encoding='utf-8'))

    results = {
        'k': K,
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
        dev_pp, dev_tok = model.perplexity(n, dev, K)
        test_pp, test_tok = model.perplexity(n, test, K)
        laplace = laplace_results['perplexity'][ORDER_NAMES[n]]
        results['perplexity'][ORDER_NAMES[n]] = {
            'order': n,
            'add_k_dev_perplexity': dev_pp,
            'add_k_test_perplexity': test_pp,
            'laplace_dev_perplexity': laplace['dev_perplexity'],
            'laplace_test_perplexity': laplace['test_perplexity'],
            'scored_tokens': dev_tok,
            'n_unique_ngrams': len(model.ngram_counts[n]),
        }
        print(f'{ORDER_NAMES[n]:>10} (n={n}): add-k(k={K}) dev PP={dev_pp:12.2f}  '
              f'test PP={test_pp:12.2f}  |  Laplace dev PP={laplace["dev_perplexity"]:12.2f}  '
              f'test PP={laplace["test_perplexity"]:12.2f}')

    # Same trigram qualitative-example contexts as Assignment 4, re-scored with add-k
    # top-5 next words (add-k only re-weights seen continuations by k, so the *ranking*
    # among attested continuations is unchanged from Laplace -- included for completeness).
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
