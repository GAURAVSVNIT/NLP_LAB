"""Naive Bayes POS tagger.

Each token is classified independently with the five features required by the assignment:

    len   : 'MORE' if len(word) > 4 else 'LESS'
    word  : the word itself
    prev  : the previous word  ('<s>' at sentence start)
    next  : the next word      ('</s>' at sentence end)
    ptag  : the previous tag   ('<s>' at sentence start)

    t* = argmax_t  log P(t) + sum_k log P(f_k | t)
    P(f_k = v | t) = (C_k(t, v) + alpha) / (C(t) + alpha * (|V_k| + 1))   (Laplace; +1 slot for unseen values)

At test time the previous-tag feature uses the tag the classifier itself predicted for the previous
token (greedy left-to-right decoding). `gold_prev_tag=True` swaps in the gold tag instead, which is
used only as an analysis of how much error propagates through that feature.
"""
from collections import Counter, defaultdict

import numpy as np

BOS, EOS = '<s>', '</s>'
FEATURES = ['len', 'word', 'prev', 'next', 'ptag']


def length_class(word):
    return 'MORE' if len(word) > 4 else 'LESS'


def token_features(words, i, prev_tag):
    return {
        'len': length_class(words[i]),
        'word': words[i],
        'prev': words[i - 1] if i > 0 else BOS,
        'next': words[i + 1] if i + 1 < len(words) else EOS,
        'ptag': prev_tag,
    }


class NaiveBayesTagger:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, sentences):
        tag_counts = Counter()
        feat_counts = {k: defaultdict(Counter) for k in FEATURES}  # feature -> value -> Counter(tag)
        for sent in sentences:
            words = [w for w, _ in sent]
            prev_tag = BOS
            for i, (_, tag) in enumerate(sent):
                tag_counts[tag] += 1
                for k, v in token_features(words, i, prev_tag).items():
                    feat_counts[k][v][tag] += 1
                prev_tag = tag

        self.tags = sorted(tag_counts)
        self.tag_index = {t: i for i, t in enumerate(self.tags)}
        n_tag = np.array([tag_counts[t] for t in self.tags], dtype=float)
        self.log_prior = np.log(n_tag / n_tag.sum())

        # Pre-compute a log-likelihood vector over tags for every seen feature value,
        # plus one shared vector for values never seen in training.
        self.log_lik, self.log_unseen = {}, {}
        for k in FEATURES:
            denom = n_tag + self.alpha * (len(feat_counts[k]) + 1)
            self.log_unseen[k] = np.log(self.alpha / denom)
            table = {}
            for v, by_tag in feat_counts[k].items():
                counts = np.array([by_tag.get(t, 0) for t in self.tags], dtype=float)
                table[v] = np.log((counts + self.alpha) / denom)
            self.log_lik[k] = table
        self.vocab = set(feat_counts['word'])
        return self

    def _score(self, feats):
        score = self.log_prior.copy()
        for k, v in feats.items():
            score += self.log_lik[k].get(v, self.log_unseen[k])
        return score

    def tag(self, words, gold_tags=None):
        out, prev_tag = [], BOS
        for i in range(len(words)):
            pred = self.tags[int(np.argmax(self._score(token_features(words, i, prev_tag))))]
            out.append(pred)
            prev_tag = gold_tags[i] if gold_tags is not None else pred
        return out
