"""Midsem Q3 — TF-IDF sentence vectors (raw-count TF, unsmoothed IDF) and the closest pair.

    tf(t, d)  = count of t in d
    idf(t)    = log10(N / df(t))
    w(t, d)   = tf(t, d) * idf(t)
    sim(a, b) = cos(v_a, v_b) = v_a . v_b / (|v_a| |v_b|)

The log base only rescales every weight by the same constant, so the cosine ranking is identical for
log10, ln or log2.
"""
import math
from itertools import combinations

SENTENCES = ['cat dog animal', 'cat cat pet', 'dog pet animal', 'cat dog pet']


def solve():
    docs = [s.split() for s in SENTENCES]
    vocab = []
    for d in docs:
        for w in d:
            if w not in vocab:
                vocab.append(w)
    N = len(docs)
    tf = [[d.count(w) for w in vocab] for d in docs]
    df = {w: sum(1 for d in docs if w in d) for w in vocab}
    idf = {w: math.log10(N / df[w]) for w in vocab}
    tfidf = [[tf[i][j] * idf[w] for j, w in enumerate(vocab)] for i in range(N)]
    norms = [math.sqrt(sum(x * x for x in v)) for v in tfidf]

    pairs = []
    for i, j in combinations(range(N), 2):
        dot = sum(a * b for a, b in zip(tfidf[i], tfidf[j]))
        cos = dot / (norms[i] * norms[j])
        euc = math.sqrt(sum((a - b) ** 2 for a, b in zip(tfidf[i], tfidf[j])))
        pairs.append({'pair': f'S{i + 1}-S{j + 1}', 'dot': dot, 'norm_a': norms[i], 'norm_b': norms[j],
                      'cosine': cos, 'euclidean': euc})
    best = max(pairs, key=lambda p: p['cosine'])
    best_euc = min(p['euclidean'] for p in pairs)
    return {
        'sentences': SENTENCES, 'vocab': vocab, 'tf': tf, 'df': df, 'idf': idf, 'tfidf': tfidf,
        'norms': norms, 'pairs': pairs, 'closest': best['pair'], 'closest_cosine': best['cosine'],
        'euclidean_ties': [p['pair'] for p in pairs if abs(p['euclidean'] - best_euc) < 1e-12],
    }


if __name__ == '__main__':
    out = solve()
    print('idf', out['idf'])
    for i, v in enumerate(out['tfidf']):
        print(f'S{i + 1}', [round(x, 4) for x in v], round(out['norms'][i], 4))
    for p in out['pairs']:
        print(p['pair'], round(p['dot'], 4), round(p['cosine'], 4), round(p['euclidean'], 4))
    print('closest', out['closest'], 'euclidean ties', out['euclidean_ties'])
