"""N-gram language model count collector and vocabulary builder for Assignment 6.

Builds vocabulary with min_count=2, mapping singletons to <UNK>.
Reuses the exact split and vocabulary specifications from Assignments 4 & 5
to allow direct comparisons across all labs.
Optimized for high performance and minimal memory footprint.
"""
import math
from collections import Counter, defaultdict
from pathlib import Path

MAX_ORDER = 4
BOS = '<s>'
EOS = '</s>'
UNK = '<UNK>'


def read_sentences(path: Path) -> list[list[str]]:
    """Read sentences from a text file, splitting by whitespace."""
    lines = path.read_text(encoding='utf-8').splitlines()
    return [line.split() for line in lines if line.strip()]


class NgramCounts:
    """Collects and stores n-gram counts, context counts, and frequency of frequencies."""
    def __init__(self, orders=(1, 2, 3, 4)):
        self.orders = orders
        self.vocab = set()
        self.V = 0
        self.ngram_counts = {n: Counter() for n in orders}
        self.context_counts = {n: Counter() for n in orders}
        self.total_tokens = 0
        
        # Track seen continuations ONLY for contexts that appear in dev/test sets
        # to avoid storing millions of set objects for unseen contexts.
        self.eval_contexts = {n: set() for n in orders}
        self.eval_context_words = {n: defaultdict(set) for n in orders}
        
        # Frequency of frequencies: freq_of_freq[n][r] = number of n-grams with frequency r
        self.freq_of_freq = {}
        
        # Continuation counts for lower-order Kneser-Ney
        self.unigram_cont_counts = Counter()
        self.total_unique_bigrams = 0

    def build_vocab(self, train_sentences: list[list[str]], min_count: int = 2) -> Counter:
        """Build vocabulary from training split, keeping words with frequency >= min_count."""
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

    def register_eval_sentences(self, eval_sentences: list[list[str]]):
        """Pre-registers all contexts appearing in dev/test sentences so that
        we only maintain continuation sets for queried contexts.
        """
        pad = MAX_ORDER - 1
        for sentence in eval_sentences:
            seq = self._padded(sentence)
            for i in range(pad, len(seq)):
                for n in self.orders:
                    ngram = tuple(seq[i - n + 1:i + 1])
                    context = ngram[:-1]
                    self.eval_contexts[n].add(context)

    def fit(self, train_sentences: list[list[str]]):
        """Count n-grams and contexts across the training corpus."""
        pad = MAX_ORDER - 1
        eval_ctx_2 = self.eval_contexts[2]
        eval_ctx_3 = self.eval_contexts[3]
        eval_ctx_4 = self.eval_contexts[4]
        
        c_words_2 = self.eval_context_words[2]
        c_words_3 = self.eval_context_words[3]
        c_words_4 = self.eval_context_words[4]

        ngram_c_1 = self.ngram_counts[1]
        ngram_c_2 = self.ngram_counts[2]
        ngram_c_3 = self.ngram_counts[3]
        ngram_c_4 = self.ngram_counts[4]

        ctx_c_1 = self.context_counts[1]
        ctx_c_2 = self.context_counts[2]
        ctx_c_3 = self.context_counts[3]
        ctx_c_4 = self.context_counts[4]

        for sentence in train_sentences:
            seq = self._padded(sentence)
            for i in range(pad, len(seq)):
                w = seq[i]
                w_1 = seq[i - 1]
                w_2 = seq[i - 2]
                w_3 = seq[i - 3]

                # Order 1
                g1 = (w,)
                c1 = ()
                ngram_c_1[g1] += 1
                ctx_c_1[c1] += 1

                # Order 2
                g2 = (w_1, w)
                c2 = (w_1,)
                ngram_c_2[g2] += 1
                ctx_c_2[c2] += 1
                if c2 in eval_ctx_2:
                    c_words_2[c2].add(w)

                # Order 3
                g3 = (w_2, w_1, w)
                c3 = (w_2, w_1)
                ngram_c_3[g3] += 1
                ctx_c_3[c3] += 1
                if c3 in eval_ctx_3:
                    c_words_3[c3].add(w)

                # Order 4
                g4 = (w_3, w_2, w_1, w)
                c4 = (w_3, w_2, w_1)
                ngram_c_4[g4] += 1
                ctx_c_4[c4] += 1
                if c4 in eval_ctx_4:
                    c_words_4[c4].add(w)

            self.total_tokens += len(seq) - pad

        # Frequency of frequencies (takes ~1s)
        for n in self.orders:
            self.freq_of_freq[n] = Counter(self.ngram_counts[n].values())

        # Unigram continuation counts: how many unique preceding words complete into w
        # i.e., number of distinct (u, w) pairs in ngram_counts[2]
        self.total_unique_bigrams = len(ngram_c_2)
        for u, w in ngram_c_2.keys():
            self.unigram_cont_counts[w] += 1

    def oov_rate(self, sentences: list[list[str]]) -> float:
        total = sum(len(s) for s in sentences)
        oov = sum(1 for s in sentences for w in s if w not in self.vocab)
        return oov / total if total else 0.0
