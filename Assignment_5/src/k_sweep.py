"""Sweep the Add-k smoothing constant across a range of values, per n-gram
order, and pick whichever k minimizes DEV perplexity for that order.

Counting (the expensive part, ~100s) happens once; the counts don't depend
on k, so re-scoring at a new k is just re-running perplexity() over the
1,000-sentence dev set, which is cheap. Test is only touched *after* the
best k is picked, and only to report how that choice generalizes -- never
to choose k itself (that would leak the held-out set into a modeling
decision).
"""
import json
import time
from pathlib import Path

from ngram_model import NgramLanguageModel, read_sentences

ROOT = Path(__file__).resolve().parent.parent
ASSIGNMENT_4_OUTPUT = ROOT.parent / 'Assignment_4' / 'output'
OUTPUT_DIR = ROOT / 'output'
ORDER_NAMES = {1: 'unigram', 2: 'bigram', 3: 'trigram', 4: 'quadrigram'}
K_VALUES = [0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.03,
            0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]


def refine(model, n, dev, coarse_points):
    """Given a coarse (k, dev_perplexity) sweep, do a second, denser
    log-spaced pass around its minimum to pin the optimum down more
    precisely, without re-counting n-grams."""
    best_k = min(coarse_points, key=lambda p: p['dev_perplexity'])['k']
    tested = {round(p['k'], 12) for p in coarse_points}
    lo, hi = best_k / 4, best_k * 4
    extra = []
    steps = 8
    for i in range(steps + 1):
        # log-spaced grid between lo and hi
        frac = i / steps
        k = lo * (hi / lo) ** frac
        k = float(f'{k:.6g}')
        if round(k, 12) in tested or k <= 0:
            continue
        tested.add(round(k, 12))
        dev_pp, _ = model.perplexity(n, dev, k)
        extra.append({'k': k, 'dev_perplexity': dev_pp})
    return coarse_points + extra


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    train = read_sentences(ASSIGNMENT_4_OUTPUT / 'train.txt')
    dev = read_sentences(ASSIGNMENT_4_OUTPUT / 'dev.txt')
    test = read_sentences(ASSIGNMENT_4_OUTPUT / 'test.txt')

    model = NgramLanguageModel(orders=(1, 2, 3, 4))
    model.build_vocab(train, min_count=2)
    model.fit(train)
    print(f'Counted n-grams over {len(train):,} sentences in {time.time()-t0:.1f}s. '
          f'Sweeping {len(K_VALUES)} k values x 4 orders on dev...')

    sweep = {ORDER_NAMES[n]: [] for n in (1, 2, 3, 4)}
    for n in (1, 2, 3, 4):
        points = []
        for k in K_VALUES:
            dev_pp, _ = model.perplexity(n, dev, k)
            points.append({'k': k, 'dev_perplexity': dev_pp})
        points = refine(model, n, dev, points)
        points.sort(key=lambda p: p['k'])
        sweep[ORDER_NAMES[n]] = points
        print(f'  {ORDER_NAMES[n]} swept + refined ({time.time()-t0:.1f}s so far)')

    best = {}
    for order, points in sweep.items():
        best[order] = min(points, key=lambda p: p['dev_perplexity']).copy()

    # Only now touch test -- to report how the dev-selected k generalizes.
    for n in (1, 2, 3, 4):
        order = ORDER_NAMES[n]
        k_best = best[order]['k']
        test_pp, _ = model.perplexity(n, test, k_best)
        best[order]['test_perplexity_at_best_k'] = test_pp
        # For reference, also test perplexity at the assignment-specified k=0.3
        assignment_k_test_pp, _ = model.perplexity(n, test, 0.3)
        best[order]['test_perplexity_at_k_0.3'] = assignment_k_test_pp

    results = {'k_values': K_VALUES, 'sweep': sweep, 'best_by_dev_perplexity': best}
    out_path = OUTPUT_DIR / 'k_sweep.json'
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\nBest k per order (chosen by minimizing DEV perplexity):')
    for order, b in best.items():
        print(f'  {order:>10}: k*={b["k"]:<6}  dev PP={b["dev_perplexity"]:10.2f}  '
              f'test PP @ k*={b["test_perplexity_at_best_k"]:10.2f}  '
              f'(vs test PP @ k=0.3: {b["test_perplexity_at_k_0.3"]:10.2f})')
    print(f'\nWrote {out_path}  (total time {time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
