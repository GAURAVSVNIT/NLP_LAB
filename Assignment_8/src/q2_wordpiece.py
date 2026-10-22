"""Midsem Q2 — WordPiece training (2 merges) and tokenisation of 'newest'.

WordPiece (as in BERT / HuggingFace) splits each word into characters. The first character is kept
as is and every later character gets the continuation prefix '##'. At each step it merges the
adjacent pair with the highest likelihood-gain score

    score(a, b) = freq(ab) / (freq(a) * freq(b))

rather than the raw pair frequency used by BPE. A new word is tokenised greedily, longest match first,
from the left. Continuation pieces must exist in the vocabulary with the '##' prefix.
"""
from collections import Counter

CORPUS = """low lower low lower lowest
low lowest low lowest
low new
new new newer newer
new"""
N_MERGES = 2
QUERY = 'newest'


def initial_split(word):
    return [word[0]] + ['##' + c for c in word[1:]]


def merge_tokens(a, b):
    return a + (b[2:] if b.startswith('##') else b)


def pair_scores(splits, word_freq):
    tok_freq, pair_freq = Counter(), Counter()
    for w, toks in splits.items():
        for t in toks:
            tok_freq[t] += word_freq[w]
        for a, b in zip(toks, toks[1:]):
            pair_freq[(a, b)] += word_freq[w]
    scores = {p: f / (tok_freq[p[0]] * tok_freq[p[1]]) for p, f in pair_freq.items()}
    return tok_freq, pair_freq, scores


def train(corpus=CORPUS, n_merges=N_MERGES):
    word_freq = Counter(corpus.split())
    splits = {w: initial_split(w) for w in word_freq}
    vocab = sorted({t for toks in splits.values() for t in toks})
    steps = []
    for step in range(1, n_merges + 1):
        tok_freq, pair_freq, scores = pair_scores(splits, word_freq)
        best_score = max(scores.values())
        # Ties are broken by first occurrence in the corpus (dict insertion order).
        best = next(p for p, s in scores.items() if s == best_score)
        new_tok = merge_tokens(*best)
        steps.append({
            'step': step,
            'token_freq': dict(tok_freq),
            'pairs': [{'pair': f'{a} + {b}', 'pair_freq': pair_freq[(a, b)], 'freq_a': tok_freq[a],
                       'freq_b': tok_freq[b], 'score': s}
                      for (a, b), s in sorted(scores.items(), key=lambda x: -x[1])],
            'merged': f'{best[0]} + {best[1]} -> {new_tok}',
            'ties': [f'{a} + {b}' for (a, b), s in scores.items() if s == best_score and (a, b) != best],
        })
        for w, toks in splits.items():
            out, i = [], 0
            while i < len(toks):
                if i + 1 < len(toks) and (toks[i], toks[i + 1]) == best:
                    out.append(new_tok)
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            splits[w] = out
        vocab.append(new_tok)
        steps[-1]['splits_after'] = {w: ' '.join(t) for w, t in splits.items()}
    return {'word_freq': dict(word_freq), 'initial_splits': {w: ' '.join(initial_split(w)) for w in word_freq},
            'steps': steps, 'vocab': vocab}


def tokenize(word, vocab):
    vocab = set(vocab)
    tokens, start, trace = [], 0, []
    while start < len(word):
        end = len(word)
        piece = None
        while end > start:
            cand = word[start:end] if start == 0 else '##' + word[start:end]
            if cand in vocab:
                piece = cand
                break
            end -= 1
        if piece is None:
            return ['[UNK]'], trace
        trace.append(f'remaining "{word[start:]}": longest vocabulary match = {piece}')
        tokens.append(piece)
        start = end
    return tokens, trace


def solve():
    model = train()
    tokens, trace = tokenize(QUERY, model['vocab'])
    model.update({'query': QUERY, 'tokens': tokens, 'trace': trace})
    return model


if __name__ == '__main__':
    out = solve()
    print('word freq:', out['word_freq'])
    for s in out['steps']:
        print(f"step {s['step']}: {s['merged']}  ties={s['ties']}")
        for p in s['pairs'][:5]:
            print('   ', p)
    print('vocab:', out['vocab'])
    print(out['query'], '->', out['tokens'])
