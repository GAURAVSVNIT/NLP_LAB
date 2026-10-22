"""Midsem Q4 — Bernoulli Naive Bayes spam classifier, prediction for E8 = (free=0, win=0, offer=0).

    P(c | x) ∝ P(c) * Π_k P(x_k | c),   P(x_k = 1 | c) = (N_{k,c} + a) / (N_c + 2a)

Absent features contribute P(x_k = 0 | c) = 1 - P(x_k = 1 | c). Results are given for the
unsmoothed MLE (a = 0, the exam's expected method) and with Laplace smoothing (a = 1).
"""
from fractions import Fraction

FEATURES = ['free', 'win', 'offer']
DATA = [
    ('E1', (1, 1, 1), 'Spam'), ('E2', (1, 1, 0), 'Spam'), ('E3', (1, 0, 1), 'Spam'), ('E4', (0, 1, 1), 'Spam'),
    ('E5', (0, 0, 1), 'NotSpam'), ('E6', (0, 1, 0), 'NotSpam'), ('E7', (1, 0, 0), 'NotSpam'),
]
QUERY = (0, 0, 0)
CLASSES = ['Spam', 'NotSpam']


def classify(alpha):
    out = {}
    n = len(DATA)
    for c in CLASSES:
        rows = [x for _, x, y in DATA if y == c]
        prior = Fraction(len(rows), n)
        terms = []
        joint = prior
        for k, f in enumerate(FEATURES):
            ones = sum(x[k] for x in rows)
            p1 = Fraction(ones + alpha, len(rows) + 2 * alpha)
            p = p1 if QUERY[k] == 1 else 1 - p1
            terms.append({'feature': f, 'count_1': ones, 'n_class': len(rows), 'p_x1': p1,
                          'value': QUERY[k], 'p': p})
            joint *= p
        out[c] = {'prior': prior, 'terms': terms, 'joint': joint}
    z = sum(v['joint'] for v in out.values())
    for v in out.values():
        v['posterior'] = v['joint'] / z
    pred = max(CLASSES, key=lambda c: out[c]['joint'])
    return out, pred


def _plain(v):
    return {'fraction': str(v), 'value': float(v)}


def solve():
    result = {}
    for name, a in [('mle', 0), ('laplace', 1)]:
        res, pred = classify(a)
        result[name] = {
            'alpha': a, 'prediction': pred,
            'classes': {c: {'prior': _plain(v['prior']), 'joint': _plain(v['joint']),
                            'posterior': _plain(v['posterior']),
                            'terms': [{**t, 'p_x1': _plain(t['p_x1']), 'p': _plain(t['p'])} for t in v['terms']]}
                        for c, v in res.items()},
        }
    return result


if __name__ == '__main__':
    for name, r in solve().items():
        print(name, '->', r['prediction'])
        for c, v in r['classes'].items():
            print('  ', c, 'prior', v['prior']['fraction'], 'terms',
                  [t['p']['fraction'] for t in v['terms']], 'joint', v['joint']['fraction'],
                  f"({v['joint']['value']:.5f})", 'posterior', round(v['posterior']['value'], 4))
