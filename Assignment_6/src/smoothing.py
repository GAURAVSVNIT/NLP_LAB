"""Implementation of the 5 advanced smoothing techniques for n-gram language models:
1. Interpolated Smoothing (Jelinek-Mercer)
2. Good-Turing Smoothing
3. Katz Backoff Smoothing
4. Stupid Backoff Smoothing (Brants et al. 2007)
5. Kneser-Ney Smoothing (Interpolated Kneser-Ney)
"""
import math
from typing import Optional

from ngram_model import NgramCounts, MAX_ORDER


def compute_gt_discount(counts: NgramCounts, n: int, k: int = 5) -> tuple[dict[int, float], float]:
    """Computes Good-Turing discount ratios d_r = r* / r for r in 1..k,
    using Katz's smooth cutoff formula.
    """
    N_r = counts.freq_of_freq[n]
    N = sum(r * c for r, c in N_r.items())
    N1 = N_r.get(1, 0)
    Nk1 = N_r.get(k + 1, 0)
    
    p0 = N1 / N if N > 0 else 1e-10
    
    d = {}
    denom = 1.0 - ((k + 1) * Nk1) / N1 if N1 > 0 else 1.0
    if denom <= 0 or denom > 2.0:
        denom = 1.0
        
    for r in range(1, k + 1):
        Nr = N_r.get(r, 0)
        Nr1 = N_r.get(r + 1, 0)
        if Nr > 0 and N1 > 0:
            numerator = (r + 1) * Nr1 / Nr - (r * (k + 1) * Nk1) / N1
            r_star = numerator / denom
            d_r = r_star / r
            if d_r <= 0.0 or d_r > 1.5:
                d_r = max(0.01, min(1.0, 1.0 - (N1 / (N1 + 2 * N_r.get(2, 1)))))
            d[r] = d_r
        else:
            d[r] = 1.0
            
    return d, p0


class InterpolatedSmoothing:
    """Jelinek-Mercer recursive linear interpolation:
    P_interp(w_i | context) = lambda_n * P_MLE(w_i | context) + (1 - lambda_n) * P_interp(w_i | context_lower)
    """
    def __init__(self, counts: NgramCounts, lambdas: Optional[dict[int, float]] = None):
        self.counts = counts
        self.V = counts.V
        self.N = counts.total_tokens
        self.lambdas = lambdas or {1: 0.999, 2: 0.65, 3: 0.50, 4: 0.40}

    def prob(self, ngram: tuple, order: int) -> float:
        if order == 1:
            w = ngram[-1]
            c_w = self.counts.ngram_counts[1].get((w,), 0)
            p_mle = c_w / self.N if self.N > 0 else 1.0 / self.V
            l1 = self.lambdas.get(1, 0.999)
            return l1 * p_mle + (1.0 - l1) * (1.0 / self.V)

        ctx = ngram[:-1]
        w = ngram[-1]
        c_ngram = self.counts.ngram_counts[order].get(ngram, 0)
        c_ctx = self.counts.context_counts[order].get(ctx, 0)
        
        p_mle = (c_ngram / c_ctx) if c_ctx > 0 else 0.0
        lam = self.lambdas.get(order, 0.5)
        
        lower_ngram = ngram[1:]
        p_lower = self.prob(lower_ngram, order - 1)
        return lam * p_mle + (1.0 - lam) * p_lower


class GoodTuringSmoothing:
    """Classical Good-Turing discounting with Katz cutoff (k=5).
    For seen n-grams (r > 0): P_GT = r* / N
    For unseen n-grams (r = 0): P_GT = (N_1 / N) / N_0
    where N_0 = V^n - seen_ngrams
    """
    def __init__(self, counts: NgramCounts, k: int = 5):
        self.counts = counts
        self.k = k
        self.V = counts.V
        self.discounts = {}
        self.p0 = {}
        self.log_p_unseen = {}
        self.total_N = {}

        for n in counts.orders:
            d, p0 = compute_gt_discount(counts, n, k=k)
            self.discounts[n] = d
            self.p0[n] = p0
            N = sum(r * c for r, c in counts.freq_of_freq[n].items())
            self.total_N[n] = N
            
            seen_types = len(counts.ngram_counts[n])
            if n == 1:
                n0 = max(1, self.V - seen_types)
                log_n0 = math.log(n0)
            else:
                log_n0 = n * math.log(self.V)
            
            self.log_p_unseen[n] = math.log(max(1e-20, p0)) - log_n0

    def prob(self, ngram: tuple, order: int) -> float:
        r = self.counts.ngram_counts[order].get(ngram, 0)
        if r == 0:
            return math.exp(self.log_p_unseen[order])
        N = self.total_N[order]
        if r > self.k:
            r_star = r
        else:
            d_r = self.discounts[order].get(r, 1.0)
            r_star = d_r * r
        return max(1e-30, r_star / N)

    def log_prob(self, ngram: tuple, order: int) -> float:
        r = self.counts.ngram_counts[order].get(ngram, 0)
        if r == 0:
            return self.log_p_unseen[order]
        N = self.total_N[order]
        if r > self.k:
            r_star = r
        else:
            d_r = self.discounts[order].get(r, 1.0)
            r_star = d_r * r
        return math.log(max(1e-30, r_star / N))


class KatzBackoffSmoothing:
    """Katz Backoff Smoothing:
    Good-Turing discounting for observed counts + recursive backoff to lower-order
    distributions with normalizer alpha(context).
    """
    def __init__(self, counts: NgramCounts, k: int = 5):
        self.counts = counts
        self.k = k
        self.V = counts.V
        self.discounts = {}
        self.p0 = {}
        self.alpha_cache = {n: {} for n in counts.orders}
        self.unigram_probs = {}

        for n in counts.orders:
            d, p0 = compute_gt_discount(counts, n, k=k)
            self.discounts[n] = d
            self.p0[n] = p0

        # Unigram base distribution
        N1 = counts.total_tokens
        u_discounts = self.discounts[1]
        sum_c_star = 0.0
        c_star_dict = {}
        for (w,), c in counts.ngram_counts[1].items():
            d_r = u_discounts.get(c, 1.0) if c <= k else 1.0
            cs = d_r * c
            c_star_dict[w] = cs
            sum_c_star += cs
        
        for w in counts.vocab:
            cs = c_star_dict.get(w, 0.5)
            self.unigram_probs[w] = cs / sum_c_star if sum_c_star > 0 else 1.0 / self.V

    def _get_alpha(self, ctx: tuple, order: int) -> float:
        """Compute and cache alpha(context) normalizer."""
        if ctx in self.alpha_cache[order]:
            return self.alpha_cache[order][ctx]
        
        c_ctx = self.counts.context_counts[order].get(ctx, 0)
        if c_ctx == 0:
            self.alpha_cache[order][ctx] = 1.0
            return 1.0
        
        seen_words = self.counts.eval_context_words[order].get(ctx, set())
        d_dict = self.discounts[order]
        
        sum_discounted_mle = 0.0
        sum_lower_prob = 0.0
        lower_ctx = ctx[1:]
        
        for w in seen_words:
            ngram = ctx + (w,)
            c_w = self.counts.ngram_counts[order].get(ngram, 0)
            d_r = d_dict.get(c_w, 1.0) if c_w <= self.k else 1.0
            sum_discounted_mle += (d_r * c_w) / c_ctx
            sum_lower_prob += self.prob(lower_ctx + (w,), order - 1)
        
        beta = max(0.0, 1.0 - sum_discounted_mle)
        denom = 1.0 - sum_lower_prob
        
        if denom > 1e-9 and beta > 0.0:
            alpha = beta / denom
        else:
            alpha = 1.0
        
        self.alpha_cache[order][ctx] = alpha
        return alpha

    def prob(self, ngram: tuple, order: int) -> float:
        if order == 1:
            w = ngram[-1]
            return self.unigram_probs.get(w, 1.0 / self.V)
        
        c_ngram = self.counts.ngram_counts[order].get(ngram, 0)
        ctx = ngram[:-1]
        c_ctx = self.counts.context_counts[order].get(ctx, 0)
        
        if c_ngram > 0 and c_ctx > 0:
            d_r = self.discounts[order].get(c_ngram, 1.0) if c_ngram <= self.k else 1.0
            return (d_r * c_ngram) / c_ctx
        
        alpha = self._get_alpha(ctx, order)
        lower_ngram = ngram[1:]
        return max(1e-30, alpha * self.prob(lower_ngram, order - 1))


class StupidBackoffSmoothing:
    """Stupid Backoff (Brants et al. 2007):
    S(w_i | context) = C(context, w_i) / C(context) if C(context, w_i) > 0
                       else alpha * S(w_i | context_lower)
    Default alpha = 0.4.
    """
    def __init__(self, counts: NgramCounts, alpha: float = 0.4):
        self.counts = counts
        self.alpha = alpha
        self.V = counts.V
        self.N = counts.total_tokens

    def score(self, ngram: tuple, order: int) -> float:
        if order == 1:
            w = ngram[-1]
            c = self.counts.ngram_counts[1].get((w,), 0)
            return max(1e-30, c / self.N) if self.N > 0 else 1.0 / self.V
        
        c_ngram = self.counts.ngram_counts[order].get(ngram, 0)
        ctx = ngram[:-1]
        c_ctx = self.counts.context_counts[order].get(ctx, 0)
        
        if c_ngram > 0 and c_ctx > 0:
            return c_ngram / c_ctx
        
        lower_ngram = ngram[1:]
        return self.alpha * self.score(lower_ngram, order - 1)

    def prob(self, ngram: tuple, order: int) -> float:
        return self.score(ngram, order)


class KneserNeySmoothing:
    """Interpolated Kneser-Ney Smoothing (Chen & Goodman 1996/1999):
    Uses absolute discounting d = N_1 / (N_1 + 2*N_2) and continuation counts
    for unigram base distribution.
    """
    def __init__(self, counts: NgramCounts, discount: Optional[float] = None):
        self.counts = counts
        self.V = counts.V
        
        # Calculate optimal discount d from Chen & Goodman: d = N1 / (N1 + 2*N2)
        if discount is not None:
            self.d = discount
        else:
            N1 = counts.freq_of_freq[2].get(1, 0)
            N2 = counts.freq_of_freq[2].get(2, 0)
            if N1 > 0 and (N1 + 2 * N2) > 0:
                self.d = N1 / (N1 + 2 * N2)
            else:
                self.d = 0.75
        self.d = max(0.05, min(0.95, self.d))
        
        # Unigram continuation base distribution
        tot_bg = counts.total_unique_bigrams
        self.unigram_cont_prob = {}
        for w, count in counts.unigram_cont_counts.items():
            self.unigram_cont_prob[w] = count / tot_bg if tot_bg > 0 else 1.0 / self.V

    def prob(self, ngram: tuple, order: int) -> float:
        if order == 1:
            w = ngram[-1]
            return self.unigram_cont_prob.get(w, 1.0 / self.V)

        ctx = ngram[:-1]
        w = ngram[-1]
        lower_ngram = ngram[1:]

        c_ngram = self.counts.ngram_counts[order].get(ngram, 0)
        c_ctx = self.counts.context_counts[order].get(ctx, 0)
        
        # Continuation set size for context
        distinct_words = len(self.counts.eval_context_words[order].get(ctx, ()))

        p_lower = self.prob(lower_ngram, order - 1)

        if c_ctx > 0:
            first_term = max(c_ngram - self.d, 0.0) / c_ctx
            lam = (self.d / c_ctx) * distinct_words
            return first_term + lam * p_lower
        else:
            return p_lower


def evaluate_perplexity(model, sentences: list[list[str]], order: int, counts: NgramCounts) -> tuple[float, int]:
    """Calculates perplexity on token sequence using the specified model and order."""
    pad = MAX_ORDER - 1
    log_prob_sum = 0.0
    token_count = 0

    has_log_prob = hasattr(model, 'log_prob')

    for sentence in sentences:
        seq = counts._padded(sentence)
        for i in range(pad, len(seq)):
            ngram = tuple(seq[i - order + 1:i + 1])
            if has_log_prob:
                lp = model.log_prob(ngram, order)
            else:
                p = model.prob(ngram, order)
                lp = math.log(max(1e-35, p))
            log_prob_sum += lp
            token_count += 1

    avg_neg_log = -log_prob_sum / token_count if token_count > 0 else 0.0
    return math.exp(avg_neg_log), token_count
