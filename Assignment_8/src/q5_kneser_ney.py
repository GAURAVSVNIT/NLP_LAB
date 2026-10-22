"""Midsem Q5 — Interpolated Kneser-Ney bigram probabilities with absolute discount d = 0.5.

    P_KN(w | v) = max(C(v w) - d, 0) / C(v)  +  lambda(v) * P_cont(w)
    lambda(v)   = d / C(v) * |{w' : C(v w') > 0}|
    P_cont(w)   = |{v' : C(v' w) > 0}| / |{(v', w') : C(v' w') > 0}|

C(v) is the number of bigram tokens that start with v. No sentence boundary markers are added.
"""
from collections import Counter
from fractions import Fraction

SENTENCES = ['I like NLP', 'I like AI', 'You like NLP', 'You love AI', 'We love NLP']
D = Fraction(1, 2)
QUERIES = [('love', 'AI'), ('like', 'NLP')]


def solve():
    bigrams = Counter()
    for s in SENTENCES:
        toks = s.split()
        bigrams.update(zip(toks, toks[1:]))
    total_types = len(bigrams)
    answers = []
    for v, w in QUERIES:
        c_v = sum(c for (a, _), c in bigrams.items() if a == v)
        c_vw = bigrams[(v, w)]
        followers = sorted({b for (a, b) in bigrams if a == v})
        preceders = sorted({a for (a, b) in bigrams if b == w})
        discounted = max(Fraction(c_vw) - D, 0) / c_v
        lam = D / c_v * len(followers)
        p_cont = Fraction(len(preceders), total_types)
        p = discounted + lam * p_cont
        answers.append({
            'query': f'P_KN({w} | {v})', 'C(v w)': c_vw, 'C(v)': c_v,
            'followers_of_v': followers, 'preceders_of_w': preceders,
            'discounted_term': str(discounted), 'lambda': str(lam), 'p_cont': str(p_cont),
            'probability': str(p), 'probability_float': float(p),
        })
    return {
        'sentences': SENTENCES, 'd': str(D),
        'bigram_counts': {f'{a} {b}': c for (a, b), c in bigrams.items()},
        'bigram_types': total_types, 'answers': answers,
    }


if __name__ == '__main__':
    out = solve()
    print(out['bigram_counts'], 'types =', out['bigram_types'])
    for a in out['answers']:
        print(a)
