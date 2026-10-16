"""First-order HMM POS tagger with three Viterbi-style decoders.

Parameters (estimated from the training data, add-alpha smoothed transitions):
  A[s, t]   = P(t_i = t | t_{i-1} = s)           forward transitions
  pi[t]     = P(t_1 = t | START)
  Bk[t, s]  = P(t_i = s | t_{i+1} = t)           reverse (backward) transitions
  end[t]    = P(t_n = t | END)
  E[w, t]   = P(w | t)                           emissions (MLE)

Words seen only once in training are replaced by UNK, so UNK learns the
emission profile of rare/unknown words. Unseen test words are mapped to UNK.

Decoders:
  forward_viterbi          left-to-right Viterbi over A
  backward_viterbi         right-to-left Viterbi over the reverse model Bk
  forward_backward_viterbi max-marginal decoding: a forward and a backward
                           max-product pass; each token gets the tag on the
                           best full-sequence path that passes through it
"""
from collections import Counter

import numpy as np

from data import UNK


class HMMTagger:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, sentences):
        word_count = Counter(w for sent in sentences for w, _ in sent)
        self.vocab = {w for w, c in word_count.items() if c > 1}
        self.tags = sorted({t for sent in sentences for _, t in sent})
        self.tag_idx = {t: i for i, t in enumerate(self.tags)}
        T = len(self.tags)

        # Count tables
        tag_count = np.zeros(T)
        start_count = np.zeros(T)
        end_count = np.zeros(T)
        bigram = np.zeros((T, T))            # bigram[s, t] = count(s followed by t)
        emit_words = sorted(self.vocab) + [UNK]
        self.word_idx = {w: i for i, w in enumerate(emit_words)}
        emit_count = np.zeros((len(emit_words), T))

        for sent in sentences:
            tags = [self.tag_idx[t] for _, t in sent]
            words = [self._norm(w) for w, _ in sent]
            start_count[tags[0]] += 1
            end_count[tags[-1]] += 1
            for i, (w, t) in enumerate(zip(words, tags)):
                tag_count[t] += 1
                emit_count[self.word_idx[w], t] += 1
                if i > 0:
                    bigram[tags[i - 1], t] += 1

        a = self.alpha
        # Forward model: row s -> column t
        self.A = np.log((bigram + a) / (bigram + a).sum(axis=1, keepdims=True))
        self.log_pi = np.log((start_count + a) / (start_count + a).sum())
        # Reverse model: Bk[t, s] = P(s | t), normalised over the earlier tag s
        col = (bigram + a) / (bigram + a).sum(axis=0, keepdims=True)
        self.Bk = np.log(col.T)
        self.log_end = np.log((end_count + a) / (end_count + a).sum())
        # Emissions P(w | t) by MLE over the (UNK-normalised) training tokens
        with np.errstate(divide='ignore'):
            self.log_emit = np.log(emit_count / tag_count)
        return self

    def _norm(self, word):
        return word if word in self.vocab else UNK

    def _emissions(self, words):
        rows = [self.word_idx[self._norm(w)] for w in words]
        return self.log_emit[rows]          # shape (n, T)

    def _to_tags(self, path):
        return [self.tags[i] for i in path]

    def forward_viterbi(self, words):
        E = self._emissions(words)
        n = len(words)
        delta = self.log_pi + E[0]
        backptr = []
        for i in range(1, n):
            scores = delta[:, None] + self.A          # rows: previous tag s
            backptr.append(scores.argmax(axis=0))     # best s for each current t
            delta = scores.max(axis=0) + E[i]

        path = [0] * n
        path[n - 1] = int(delta.argmax())
        for i in range(n - 1, 0, -1):
            path[i - 1] = int(backptr[i - 1][path[i]])
        return self._to_tags(path)

    def backward_viterbi(self, words):
        E = self._emissions(words)
        n = len(words)
        delta = self.log_end + E[n - 1]
        backptr = [None] * n
        for i in range(n - 2, -1, -1):
            scores = delta[:, None] + self.Bk         # rows: later tag t, columns: earlier s
            backptr[i] = scores.argmax(axis=0)        # best later tag for each earlier s
            delta = scores.max(axis=0) + E[i]

        path = [0] * n
        path[0] = int(delta.argmax())
        for i in range(0, n - 1):
            path[i + 1] = int(backptr[i][path[i]])
        return self._to_tags(path)

    def forward_backward_viterbi(self, words):
        E = self._emissions(words)
        n = len(words)
        # F[i, t]: best score of w_0..w_{i-1} with t_i = t (emission at i excluded)
        F = np.zeros_like(E)
        F[0] = self.log_pi
        for i in range(1, n):
            scores = (F[i - 1] + E[i - 1])[:, None] + self.A
            F[i] = scores.max(axis=0)
        # K[i, t]: best score of w_{i+1}..w_{n-1} given t_i = t (emission at i excluded)
        K = np.zeros_like(E)
        for i in range(n - 2, -1, -1):
            K[i] = (self.A + (E[i + 1] + K[i + 1])[None, :]).max(axis=1)
        # Max-marginal per token: combine both passes around the emission at i
        scores = F + E + K
        return self._to_tags(scores.argmax(axis=1).tolist())
