"""First-order (bigram) Hidden Markov Model POS tagger with three decoders.

Parameters (all estimated from the training split):

    forward transitions    A_f[i, j] = P(t_k = j | t_{k-1} = i)   rows include <s>, columns include </s>
    backward transitions   A_b[i, j] = P(t_k = j | t_{k+1} = i)   rows include </s>, columns include <s>
    emissions              B[j, w]   = P(w | t = j)

Both transition tables use add-k smoothing. Emissions use the training counts, and some probability
mass under each tag is kept back for unseen words. That mass is proportional to the number of
hapax-legomena word types seen with the tag, a Good-Turing-style estimate of how likely a new word is
under that tag. Open classes (N_NNP, N_NN, JJ, ...) therefore get far more unknown-word mass than
closed classes (PSP, CC_CCD, ...).

    P(w | t) = C(t, w) / (C(t) + H_t)       for training words
    P(UNK | t) = H_t / (C(t) + H_t)         for every unseen word,  H_t = #hapax types tagged t (+0.5)

Decoders
    forward Viterbi          left-to-right max-product recursion with A_f (standard Viterbi).
    backward Viterbi         right-to-left max-product recursion over the reversed chain, starting
                             at </s> and moving towards <s> with A_b. Every tag occurrence has
                             exactly one predecessor and one successor on the augmented chain, so a
                             tag's row and column sums are equal. A_f and A_b then share their
                             numerators, and their denominators cancel over a whole sentence, so the
                             two chains assign the *same* joint probability to every (tags, words)
                             pair, for any add-k. Backward Viterbi therefore returns the same argmax
                             (train_eval.py checks this numerically).
    forward-backward Viterbi runs a full forward pass (alpha) and a full backward pass (beta) over the
                             trellis and labels every position with argmax_j alpha_t(j) * beta_t(j),
                             the tag with the highest posterior P(t_k = j | sentence). This maximises
                             the expected number of correctly tagged tokens rather than the
                             probability of the whole sequence.
All computation is done in log space.
"""
from collections import Counter

import numpy as np

BOS, EOS = '<s>', '</s>'


def _logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


class HMMTagger:
    def __init__(self, k_transition=0.1):
        self.k = k_transition

    def fit(self, sentences):
        tag_counts, word_tag = Counter(), Counter()
        for sent in sentences:
            for w, t in sent:
                tag_counts[t] += 1
                word_tag[(w, t)] += 1
        self.tags = sorted(tag_counts)
        T = len(self.tags)
        idx = {t: i for i, t in enumerate(self.tags)}
        self.tag_index = idx

        # Bigram tag counts on the augmented chain <s> t_1 ... t_n </s>: rows = previous (incl. <s>
        # at index T), columns = next (incl. </s> at index T).
        bigram = np.zeros((T + 1, T + 1))
        for sent in sentences:
            seq = [T] + [idx[t] for _, t in sent] + [T]
            for a, b in zip(seq, seq[1:]):
                bigram[a, b] += 1
        self.bigram_counts = bigram

        # Forward:  P(next | prev)  -> normalise rows.
        fwd = (bigram + self.k) / (bigram + self.k).sum(axis=1, keepdims=True)
        self.log_start = np.log(fwd[T, :T])          # P(t_1 | <s>)
        self.log_trans = np.log(fwd[:T, :T])         # P(t_k | t_{k-1})
        self.log_end = np.log(fwd[:T, T])            # P(</s> | t_n)

        # Backward:  P(prev | next)  -> the same counts read column-wise.
        rev = bigram.T
        bwd = (rev + self.k) / (rev + self.k).sum(axis=1, keepdims=True)
        self.log_bstart = np.log(bwd[T, :T])         # P(t_n | </s>)
        self.log_btrans = np.log(bwd[:T, :T])        # [i, j] = P(t_k = j | t_{k+1} = i)
        self.log_bend = np.log(bwd[:T, T])           # P(<s> | t_1)

        # Emissions with hapax-based unknown-word mass.
        word_freq = Counter()
        for (w, _), c in word_tag.items():
            word_freq[w] += c
        hapax = np.full(T, 0.5)
        for (w, t), c in word_tag.items():
            if word_freq[w] == 1:
                hapax[idx[t]] += 1
        n_tag = np.array([tag_counts[t] for t in self.tags], dtype=float)
        denom = n_tag + hapax
        self.log_unk = np.log(hapax / denom)
        emis = {}
        for (w, t), c in word_tag.items():
            emis.setdefault(w, np.zeros(T))[idx[t]] = c
        with np.errstate(divide='ignore'):
            self.log_emis = {w: np.log(v / denom) for w, v in emis.items()}
        self.vocab = set(emis)
        return self

    def _emission_matrix(self, words):
        return np.stack([self.log_emis.get(w, self.log_unk) for w in words])  # (n, T)

    # ----- decoders -----
    def forward_viterbi(self, words):
        E = self._emission_matrix(words)
        n, T = E.shape
        delta = self.log_start + E[0]
        back = np.zeros((n, T), dtype=np.int32)
        for t in range(1, n):
            scores = delta[:, None] + self.log_trans          # [prev, cur]
            back[t] = np.argmax(scores, axis=0)
            delta = scores[back[t], np.arange(T)] + E[t]
        best = int(np.argmax(delta + self.log_end))
        path = [best]
        for t in range(n - 1, 0, -1):
            best = int(back[t, best])
            path.append(best)
        return [self.tags[i] for i in reversed(path)]

    def backward_viterbi(self, words):
        E = self._emission_matrix(words)
        n, T = E.shape
        delta = self.log_bstart + E[n - 1]
        fwd_ptr = np.zeros((n, T), dtype=np.int32)          # best tag at k+1 given tag at k
        for t in range(n - 2, -1, -1):
            scores = delta[:, None] + self.log_btrans         # [next, cur]
            fwd_ptr[t] = np.argmax(scores, axis=0)
            delta = scores[fwd_ptr[t], np.arange(T)] + E[t]
        best = int(np.argmax(delta + self.log_bend))
        path = [best]
        for t in range(0, n - 1):
            best = int(fwd_ptr[t, best])
            path.append(best)
        return [self.tags[i] for i in path]

    def forward_backward_viterbi(self, words):
        E = self._emission_matrix(words)
        n, T = E.shape
        alpha = np.zeros((n, T))
        beta = np.zeros((n, T))
        alpha[0] = self.log_start + E[0]
        for t in range(1, n):
            alpha[t] = _logsumexp(alpha[t - 1][:, None] + self.log_trans, axis=0) + E[t]
        beta[n - 1] = self.log_end
        for t in range(n - 2, -1, -1):
            beta[t] = _logsumexp(self.log_trans + (E[t + 1] + beta[t + 1])[None, :], axis=1)
        posterior = alpha + beta
        return [self.tags[i] for i in np.argmax(posterior, axis=1)]

    def joint_logprob(self, words, tags, direction):
        """log P(tags, words) under the forward (left-to-right) or backward (right-to-left) chain."""
        E = self._emission_matrix(words)
        ids = [self.tag_index[t] for t in tags]
        emit = sum(E[k, j] for k, j in enumerate(ids))
        if direction == 'forward':
            trans = self.log_start[ids[0]] + self.log_end[ids[-1]]
            trans += sum(self.log_trans[a, b] for a, b in zip(ids, ids[1:]))
        else:
            trans = self.log_bstart[ids[-1]] + self.log_bend[ids[0]]
            trans += sum(self.log_btrans[b, a] for a, b in zip(ids, ids[1:]))
        return emit + trans

    def max_marginal_check(self, words):
        """Combines forward and backward *max-product* trellises of the forward model. Each position
        then picks the tag on the single best path, so this must reproduce `forward_viterbi`
        (up to exact ties). It is used only as a correctness check."""
        E = self._emission_matrix(words)
        n, T = E.shape
        f = np.zeros((n, T))
        b = np.zeros((n, T))
        f[0] = self.log_start + E[0]
        for t in range(1, n):
            f[t] = np.max(f[t - 1][:, None] + self.log_trans, axis=0) + E[t]
        b[n - 1] = self.log_end
        for t in range(n - 2, -1, -1):
            b[t] = np.max(self.log_trans + (E[t + 1] + b[t + 1])[None, :], axis=1)
        return [self.tags[i] for i in np.argmax(f + b, axis=1)]
