"""Unigram/bigram/trigram/quadrigram language models, counted once and
scored with Add-k smoothing:

P_add-k(w_i | w_{i-n+1}^{i-1}) = (C(w_{i-n+1}^{i-1}, w_i) + k) / (C(w_{i-n+1}^{i-1}) + k*V)

This is the same count-collection as Assignment 4's Laplace model (Add-k
with k=1 *is* Laplace) -- only the numerator/denominator constant changes.
Vocabulary/UNK handling is identical to Assignment 4 so the two labs are
directly comparable: words seen exactly once in training collapse to <UNK>,
and V is fixed regardless of k.
"""
import math
from collections import Counter
from pathlib import Path

MAX_ORDER = 4
BOS = '<s>'
EOS = '</s>'
UNK = '<UNK>'


def read_sentences(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding='utf-8').splitlines()
    return [line.split() for line in lines if line.strip()]


class NgramLanguageModel:
    def __init__(self, orders=(1, 2, 3, 4)):
        self.orders = orders
        self.vocab = None
        self.V = 0
        self.ngram_counts = {n: Counter() for n in orders}
        self.context_counts = {n: Counter() for n in orders}
        self.total_tokens = 0

    def build_vocab(self, train_sentences: list[list[str]], min_count: int = 2):
        freq = Counter(w for sent in train_sentences for w in sent)
        kept = {w for w, c in freq.items() if c >= min_count}
        self.vocab = kept | {UNK, EOS}
        self.V = len(self.vocab)
        return freq

    def _map(self, word: str) -> str:
        return word if word in self.vocab else UNK

    def _padded(self, sentence: list[str]) -> list[str]:
        pad = MAX_ORDER - 1
        return [BOS] * pad + [self._map(w) for w in sentence] + [EOS]

    def fit(self, train_sentences: list[list[str]]):
        pad = MAX_ORDER - 1
        for sentence in train_sentences:
            seq = self._padded(sentence)
            for i in range(pad, len(seq)):
                for n in self.orders:
                    ngram = tuple(seq[i - n + 1:i + 1])
                    context = ngram[:-1]
                    self.ngram_counts[n][ngram] += 1
                    self.context_counts[n][context] += 1
            self.total_tokens += len(seq) - pad

    def add_k_prob(self, n: int, ngram: tuple, k: float) -> float:
        context = ngram[:-1]
        c_ngram = self.ngram_counts[n].get(ngram, 0)
        c_context = self.context_counts[n].get(context, 0)
        return (c_ngram + k) / (c_context + k * self.V)

    def perplexity(self, n: int, sentences: list[list[str]], k: float) -> tuple[float, int]:
        pad = MAX_ORDER - 1
        log_prob_sum = 0.0
        token_count = 0
        for sentence in sentences:
            seq = self._padded(sentence)
            for i in range(pad, len(seq)):
                ngram = tuple(seq[i - n + 1:i + 1])
                p = self.add_k_prob(n, ngram, k)
                log_prob_sum += math.log(p)
                token_count += 1
        avg_neg_log = -log_prob_sum / token_count
        return math.exp(avg_neg_log), token_count

    def oov_rate(self, sentences: list[list[str]]) -> float:
        total = sum(len(s) for s in sentences)
        oov = sum(1 for s in sentences for w in s if w not in self.vocab)
        return oov / total if total else 0.0
