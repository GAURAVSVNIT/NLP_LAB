# Lab 6 — Advanced Smoothing Techniques for N-gram Language Models

This laboratory implements, analyzes, and evaluates five advanced smoothing techniques across four n-gram language models (unigram, bigram, trigram, and quadgram) on the Gujarati (`guj_Gujr`) corpus, tested on the held-out development and test splits from [Assignment 4](../Assignment_4) and [Assignment 5](../Assignment_5):

1. **Interpolated Smoothing (Jelinek-Mercer)** (For Bigram, Trigram, and Quadgram Models)
2. **Good-Turing Smoothing**
3. **Katz Backoff Smoothing**
4. **Stupid Backoff Smoothing**
5. **Kneser-Ney Smoothing**

---

## 1. Background: Resolving Additive Over-Smoothing

In Assignment 4 (Add-one / Laplace) and Assignment 5 (Add-k with $k=0.3$), both models suffered from a catastrophic failure mode: **as n-gram order increased, perplexity degraded exponentially**:

- **Unigram:** $\sim 2,422$ test PP
- **Bigram:** $\sim 2,726$ test PP
- **Trigram:** $\sim 23,594$ test PP
- **Quadgram:** $\sim 66,251$ test PP

### The Root Cause
The mathematical culprit is the additive denominator correction term $k \cdot |V|$. With vocabulary size $|V| = 222,199$, additive smoothing steals massive probability mass from attested n-grams and spreads it uniformly across all $|V|^n$ continuation paths. For trigrams and quadgrams, virtually every test context is unseen in training; adding $k \cdot 222,199$ to the denominator means observed continuations receive infinitesimal probabilities, while millions of non-existent continuations receive non-trivial mass.

Assignment 6 resolves this pathology by replacing flat additive constants with:
- **Count discounting** based on the frequency of frequencies ($N_r$).
- **Recursive backoff** to lower-order models for unseen events.
- **Continuation probability** that measures how versatile a word is across diverse preceding contexts rather than how globally frequent it is.

---

## 2. Experimental Results

Reusing the exact 998,000-sentence training split, vocabulary ($V = 222,199$), and held-out 1,000-sentence development and test sets from Assignments 4 and 5:

### 2.1 Test-Set Perplexity Comparison

| Model Order | Laplace (Lab 4) | Add-k ($k=0.3$, Lab 5) | Good-Turing (Standalone) | Stupid Backoff ($\alpha=0.4$) | Katz Backoff | Interpolated (J-M) | Kneser-Ney (Interpolated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Unigram** ($n=1$) | 2,429.28 | 2,422.63 | 2,420.40 | 2,420.40 | 2,420.40 | 2,420.90 | 4,046.93 |
| **Bigram** ($n=2$) | 4,896.38 | 2,725.88 | 2,400,611.51 | **373.90** | 404.73 | 474.54 | 385.44 |
| **Trigram** ($n=3$) | 34,912.75 | 23,594.46 | $2.56 \times 10^{11}$ | 441.30 | 359.97 | 432.56 | **344.27** |
| **Quadgram** ($n=4$) | 83,181.82 | 66,251.18 | $1.43 \times 10^{17}$ | 802.77 | 375.53 | 436.90 | **358.27** |

### 2.2 Development-Set Perplexity Comparison

| Model Order | Laplace (Lab 4) | Add-k ($k=0.3$, Lab 5) | Good-Turing (Standalone) | Stupid Backoff ($\alpha=0.4$) | Katz Backoff | Interpolated (J-M) | Kneser-Ney (Interpolated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Unigram** ($n=1$) | 2,610.93 | 2,605.64 | 2,604.16 | 2,604.16 | 2,604.16 | 2,604.46 | 4,240.73 |
| **Bigram** ($n=2$) | 5,359.84 | 2,976.94 | 2,695,074.10 | **394.58** | 431.17 | 500.49 | 407.83 |
| **Trigram** ($n=3$) | 37,653.83 | 25,575.86 | $3.40 \times 10^{11}$ | 462.88 | 377.44 | 452.48 | **360.95** |
| **Quadgram** ($n=4$) | 87,766.19 | 70,423.65 | $2.14 \times 10^{17}$ | 857.76 | 393.17 | 458.66 | **377.31** |

---

## 3. Visualizations

The pipeline generates four high-resolution comparison plots saved in `assets/`:

1. **`assets/order_scaling_trend.png`**:
   Visualizes the perplexity trajectory as order increases from 1 to 4. Additive models diverge exponentially to ~83,000, whereas Kneser-Ney, Katz Backoff, and Interpolation drop sharply from ~2,400 to ~344–436!
2. **`assets/kneser_ney_vs_baselines.png`**:
   Directly contrasts Kneser-Ney against Laplace and Add-k on the test set.
3. **`assets/smoothing_comparison_test.png`**:
   Grouped bar chart comparing all seven models on the test set.
4. **`assets/smoothing_comparison_dev.png`**:
   Grouped bar chart on the development set.

---

## 4. Key Findings & Analysis

1. **The Triumph of Backoff and Continuation Counts (Kneser-Ney):**
   - Kneser-Ney achieves a stunning **98.5% reduction** in test perplexity for Trigrams (from 23,594 in Add-k to **344.27**) and a **99.5% reduction** for Quadgrams (from 66,251 to **358.27**).
   - Higher-order language models finally do what they are supposed to do: **outperform unigrams and capture multi-word syntax**.
2. **Why Standalone Good-Turing Exploded at Higher Orders:**
   - Standalone Good-Turing computes the unseen probability mass $P_0 = N_1 / N$ accurately, but divides it equally across all $N_0 = |V|^n - |Seen|$ unseen n-grams.
   - For $n=3$, $|V|^3 \approx 1.1 \times 10^{16}$, so each unseen trigram receives an astronomical penalty ($\sim 10^{-17}$), driving perplexity to $2.56 \times 10^{11}$.
   - **Katz Backoff** fixes this completely: instead of uniform distribution over unseen n-grams, it distributes the discounted mass recursively into lower-order distributions weighted by $\alpha(context)$, achieving test PP of **359.97** for trigrams and **375.53** for quadgrams!
3. **Jelinek-Mercer Interpolation:**
   - Dev-tuned weights ($\lambda_2 = 0.70, \lambda_3 = 0.20, \lambda_4 = 0.05$) achieve strong test perplexities (474.54 for bigram, 432.56 for trigram, 436.90 for quadgram), verifying that blending higher-order MLE with lower-order estimates prevents context sparsity from destroying predictions.
4. **Stupid Backoff:**
   - With fixed $\alpha = 0.4$, Stupid Backoff performs remarkably well on bigrams (373.90) and trigrams (441.30) with zero parameter tuning, demonstrating why it became the industry standard for web-scale systems.

---

## 5. Deliverables & How to Run

```bash
# Install dependencies
uv sync

# Run the complete pipeline (generates charts and PDF report)
uv run src/main.py

# To force a full retrain and recount pass from scratch:
uv run src/main.py --retrain
```

### Artifacts Produced:
- [`output/results.json`](output/results.json): Full structured JSON containing vocab stats, tuned lambdas, and dev/test perplexities across all 5 techniques and 4 orders.
- [`assets/`](assets/): Generated high-resolution charts.
- [`LAB-6-Report.pdf`](LAB-6-Report.pdf): Publication-quality formatted PDF report.
