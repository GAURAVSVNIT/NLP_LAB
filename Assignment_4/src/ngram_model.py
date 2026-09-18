"""Unigram/bigram/trigram/quadrigram language models with Add-one (Laplace)
smoothing, trained together in a single pass over the corpus.

P_Laplace(w_n | w_1..w_{n-1}) = (C(w_1..w_n) + 1) / (C(w_1..w_{n-1}) + V)

Words seen exactly once in training are collapsed to <UNK> so the model has
a defined way to assign probability mass to unseen words at eval time (any
dev/test word absent from the kept vocabulary is also mapped to <UNK>).
V is the size of that vocabulary (kept words + <UNK> + </s>, following the
convention that </s> counts as a predictable token but the start pad <s>
does not, since <s> is only ever context, never a prediction target).
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

    def laplace_prob(self, n: int, ngram: tuple) -> float:
        context = ngram[:-1]
        c_ngram = self.ngram_counts[n].get(ngram, 0)
        c_context = self.context_counts[n].get(context, 0)
        return (c_ngram + 1) / (c_context + self.V)

    def perplexity(self, n: int, sentences: list[list[str]]) -> tuple[float, int]:
        pad = MAX_ORDER - 1
        log_prob_sum = 0.0
        token_count = 0
        for sentence in sentences:
            seq = self._padded(sentence)
            for i in range(pad, len(seq)):
                ngram = tuple(seq[i - n + 1:i + 1])
                p = self.laplace_prob(n, ngram)
                log_prob_sum += math.log(p)
                token_count += 1
        avg_neg_log = -log_prob_sum / token_count
        return math.exp(avg_neg_log), token_count

    def oov_rate(self, sentences: list[list[str]]) -> float:
        total = sum(len(s) for s in sentences)
        oov = sum(1 for s in sentences for w in s if w not in self.vocab)
        return oov / total if total else 0.0
