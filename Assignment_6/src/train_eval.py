"""Train and evaluate the 5 smoothing techniques across 4 n-gram orders
on the Gujarati corpus dev and test sets, comparing against Assignment 4 (Laplace)
and Assignment 5 (Add-k, k=0.3) baselines.
"""
import json
import time
from pathlib import Path

from ngram_model import NgramCounts, read_sentences
from smoothing import (
    InterpolatedSmoothing,
    GoodTuringSmoothing,
    KatzBackoffSmoothing,
    StupidBackoffSmoothing,
    KneserNeySmoothing,
    evaluate_perplexity,
)

ROOT = Path(__file__).resolve().parent.parent
ASSIGNMENT_4_OUTPUT = ROOT.parent / 'Assignment_4' / 'output'
ASSIGNMENT_5_OUTPUT = ROOT.parent / 'Assignment_5' / 'output'
OUTPUT_DIR = ROOT / 'output'
ORDER_NAMES = {1: 'unigram', 2: 'bigram', 3: 'trigram', 4: 'quadrigram'}


def tune_interpolated_lambdas(counts: NgramCounts, dev_sentences: list[list[str]]) -> dict[int, float]:
    """Finds best interpolation weights lambda on the development set."""
    print('Tuning Jelinek-Mercer interpolation weights on dev set...')
    best_lambdas = {1: 0.999}
    
    # 1. Bigram lambda: blend bigram MLE and unigram
    best_pp = float('inf')
    best_l2 = 0.65
    for l2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        model = InterpolatedSmoothing(counts, lambdas={1: 0.999, 2: l2})
        pp, _ = evaluate_perplexity(model, dev_sentences, 2, counts)
        if pp < best_pp:
            best_pp = pp
            best_l2 = l2
    best_lambdas[2] = best_l2
    print(f'  Optimal Bigram lambda_2: {best_l2} (dev PP: {best_pp:.2f})')

    # 2. Trigram lambda: blend trigram MLE and interpolated bigram
    best_pp = float('inf')
    best_l3 = 0.50
    for l3 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        model = InterpolatedSmoothing(counts, lambdas={1: 0.999, 2: best_l2, 3: l3})
        pp, _ = evaluate_perplexity(model, dev_sentences, 3, counts)
        if pp < best_pp:
            best_pp = pp
            best_l3 = l3
    best_lambdas[3] = best_l3
    print(f'  Optimal Trigram lambda_3: {best_l3} (dev PP: {best_pp:.2f})')

    # 3. Quadgram lambda: blend quadgram MLE and interpolated trigram
    best_pp = float('inf')
    best_l4 = 0.40
    for l4 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        model = InterpolatedSmoothing(counts, lambdas={1: 0.999, 2: best_l2, 3: best_l3, 4: l4})
        pp, _ = evaluate_perplexity(model, dev_sentences, 4, counts)
        if pp < best_pp:
            best_pp = pp
            best_l4 = l4
    best_lambdas[4] = best_l4
    print(f'  Optimal Quadgram lambda_4: {best_l4} (dev PP: {best_pp:.2f})')

    return best_lambdas


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print('Loading dataset splits from Assignment 4/output...')
    train = read_sentences(ASSIGNMENT_4_OUTPUT / 'train.txt')
    dev = read_sentences(ASSIGNMENT_4_OUTPUT / 'dev.txt')
    test = read_sentences(ASSIGNMENT_4_OUTPUT / 'test.txt')
    print(f'train={len(train):,}  dev={len(dev):,}  test={len(test):,} sentences.')

    counts = NgramCounts(orders=(1, 2, 3, 4))
    freq = counts.build_vocab(train, min_count=2)
    print(f'Vocabulary: V = {counts.V:,} (raw types in train: {len(freq):,})')

    counts.register_eval_sentences(dev + test)
    print('Fitting n-gram counts, frequency-of-frequencies, and continuation counts...')
    counts.fit(train)
    print(f'Done counting. Total scored tokens: {counts.total_tokens:,} ({time.time() - t0:.1f}s)')

    # Load baseline results
    laplace_results = json.loads((ASSIGNMENT_4_OUTPUT / 'results.json').read_text(encoding='utf-8'))
    addk_results = json.loads((ASSIGNMENT_5_OUTPUT / 'results.json').read_text(encoding='utf-8'))

    # Tune Interpolation lambdas on dev set
    best_lambdas = tune_interpolated_lambdas(counts, dev)

    # Initialize the 5 models
    models = {
        'interpolated': InterpolatedSmoothing(counts, lambdas=best_lambdas),
        'good_turing': GoodTuringSmoothing(counts, k=5),
        'katz_backoff': KatzBackoffSmoothing(counts, k=5),
        'stupid_backoff': StupidBackoffSmoothing(counts, alpha=0.4),
        'kneser_ney': KneserNeySmoothing(counts),
    }

    results = {
        'vocab_size': counts.V,
        'raw_train_types': len(freq),
        'train_tokens': counts.total_tokens,
        'n_train_sentences': len(train),
        'n_dev_sentences': len(dev),
        'n_test_sentences': len(test),
        'dev_oov_rate': counts.oov_rate(dev),
        'test_oov_rate': counts.oov_rate(test),
        'interpolated_lambdas': best_lambdas,
        'kneser_ney_discount': models['kneser_ney'].d,
        'methods': {},
        'by_order': {o: {} for o in ORDER_NAMES.values()},
    }

    print('\nEvaluating Perplexities across all 5 Smoothing Techniques...')
    print('=' * 85)
    print(f'{"Method":<18} | {"Order":<11} | {"Dev Perplexity":>16} | {"Test Perplexity":>16}')
    print('-' * 85)

    for method_name, model in models.items():
        results['methods'][method_name] = {'perplexity': {}}
        orders_to_eval = (2, 3, 4) if method_name == 'interpolated' else (1, 2, 3, 4)
        
        # Include order 1 for interpolated as unigram base
        if method_name == 'interpolated':
            dev_pp, dev_tok = evaluate_perplexity(model, dev, 1, counts)
            test_pp, test_tok = evaluate_perplexity(model, test, 1, counts)
            results['methods'][method_name]['perplexity']['unigram'] = {
                'order': 1, 'dev_perplexity': dev_pp, 'test_perplexity': test_pp, 'tokens': dev_tok
            }
            results['by_order']['unigram'][method_name] = {'dev': dev_pp, 'test': test_pp}

        for n in (1, 2, 3, 4) if method_name != 'interpolated' else (2, 3, 4):
            order_label = ORDER_NAMES[n]
            t_eval = time.time()
            dev_pp, dev_tok = evaluate_perplexity(model, dev, n, counts)
            test_pp, test_tok = evaluate_perplexity(model, test, n, counts)
            
            results['methods'][method_name]['perplexity'][order_label] = {
                'order': n,
                'dev_perplexity': dev_pp,
                'test_perplexity': test_pp,
                'tokens': dev_tok,
                'unique_ngrams': len(counts.ngram_counts[n]),
            }
            results['by_order'][order_label][method_name] = {'dev': dev_pp, 'test': test_pp}
            print(f'{method_name:<18} | {order_label:<11} | {dev_pp:16.2f} | {test_pp:16.2f}  ({time.time() - t_eval:.1f}s)')

    # Add baselines into by_order
    for n in (1, 2, 3, 4):
        o = ORDER_NAMES[n]
        results['by_order'][o]['laplace'] = {
            'dev': laplace_results['perplexity'][o]['dev_perplexity'],
            'test': laplace_results['perplexity'][o]['test_perplexity']
        }
        results['by_order'][o]['add_k'] = {
            'dev': addk_results['perplexity'][o]['add_k_dev_perplexity'],
            'test': addk_results['perplexity'][o]['add_k_test_perplexity']
        }

    # Qualitative comparison: top continuations for dev contexts
    context_next = {}
    for ngram, count in counts.ngram_counts[3].items():
        context_next.setdefault(ngram[:2], []).append((ngram[-1], count))

    examples = []
    for sent in dev[:5]:
        if len(sent) < 3:
            continue
        w1, w2 = sent[0], sent[1]
        ctx = tuple(counts._map(w) for w in (w1, w2))
        
        # Rank by Kneser-Ney probability
        kn_scores = []
        for cand, _ in context_next.get(ctx, []):
            trigram = ctx + (cand,)
            p = models['kneser_ney'].prob(trigram, 3)
            kn_scores.append((cand, p))
        kn_top = [w for w, _ in sorted(kn_scores, key=lambda x: x[1], reverse=True)[:5]]
        
        examples.append({
            'context': [w1, w2],
            'kneser_ney_top5': kn_top,
        })
    results['trigram_examples'] = examples

    out_file = OUTPUT_DIR / 'results.json'
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nWrote evaluation results to {out_file} (Total pipeline time: {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
