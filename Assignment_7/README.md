# Lab 7 — Hindi POS Tagging: Naive Bayes and HMM Viterbi Decoders

This lab builds a part-of-speech tagger for Hindi (BIS tagset, 24 tags) from the supplied train / dev / test files in [`Hindi-POS-Data/`](Hindi-POS-Data) and compares four algorithms:

1. **Naive Bayes classifier**. Each token is tagged independently from five features: length class (`MORE` if `len > 4` else `LESS`), the word, the previous word, the next word and the previous tag.
2. **HMM, forward Viterbi**. Standard left-to-right Viterbi over a bigram HMM.
3. **HMM, backward Viterbi**. Right-to-left Viterbi over the reversed chain, using `P(t_k | t_{k+1})`.
4. **HMM, forward-backward Viterbi**. Forward (α) and backward (β) passes over the trellis, then each token takes the tag with the highest posterior `α·β`.

All probabilities are estimated from the training split only. The dev split chooses the two smoothing constants (NB Laplace α = 0.1, HMM transition add-k = 0.1). The test split is used once, for reporting.

---

## 1. Data

| Split | Sentences | Tokens | OOV rate |
|---|---:|---:|---:|
| Train | 15,087 | 325,710 | — (18,279 word types) |
| Dev | 1,866 | 40,762 | 4.19% |
| Test | 1,910 | 41,415 | 4.59% |

## 2. Results

| Algorithm | Split | Accuracy | Macro F1 | Weighted F1 | OOV-word acc. |
|---|---|---:|---:|---:|---:|
| Naive Bayes | dev | 92.47 | 83.31 | 92.49 | 51.14 |
| HMM forward Viterbi | dev | 94.36 | 94.65 | 94.38 | 58.11 |
| HMM backward Viterbi | dev | 94.36 | 94.65 | 94.38 | 58.11 |
| HMM forward-backward Viterbi | dev | **94.39** | **94.70** | **94.40** | **58.70** |
| Naive Bayes | test | 92.57 | 84.44 | 92.59 | 55.77 |
| HMM forward Viterbi | test | **94.37** | 91.16 | **94.38** | **61.08** |
| HMM backward Viterbi | test | **94.37** | 91.16 | **94.38** | **61.08** |
| HMM forward-backward Viterbi | test | 94.31 | **91.17** | 94.32 | 60.51 |

Per-tag precision / recall / F1 for every algorithm and split are in [`output/results.json`](output/results.json), and the per-tag F1 table is in the report.

## 3. Key Findings

- **The HMM beats Naive Bayes by about 1.8 accuracy points and 7 macro-F1 points.** The gap is concentrated in rare closed-class tags (RB, PR_PRL, QT_QTO, PR_PRQ, RP_NEG, N_NST). On those tags Naive Bayes has high recall but low precision. Laplace smoothing on the open-vocabulary context-word features penalises an unseen value *less* for rare tags, so rare tags get pulled in.
- **Forward and backward Viterbi give exactly the same output.** On the `<s> … </s>` chain every tag has one predecessor and one successor, so the bigram matrix's row and column sums are equal. The left-to-right and right-to-left HMMs therefore assign the *same* joint probability to every (tags, words) pair, for any add-k. The largest log-probability difference over the dev sentences is 2.3e-13, which is floating-point rounding.
- **Forward-backward (posterior) decoding is statistically tied with Viterbi.** It changes only about 0.4% of tokens, with a slight gain on dev and a slight loss on test.
- **Error analysis.** The main errors are N_NN ↔ N_NNP (no capitalisation in Hindi) and out-of-vocabulary words: OOV accuracy is about 60%, against about 96% for known words.
- **Previous-tag error propagation in Naive Bayes is small.** Using the gold previous tag gives only +0.6 points.
- **Correctness check.** Combining the forward and backward max-product trellises reproduces the forward-Viterbi path on 100% of tokens.

## 4. Charts (`assets/`)

| File | Content |
|---|---|
| `per_tag_f1_test.png`, `per_tag_f1_dev.png` | Grouped per-tag F1 bars for all four algorithms |
| `f1_heatmap_test.png`, `f1_heatmap_dev.png` | Tag × algorithm F1 heat map |
| `overall_comparison.png` | Accuracy / macro-F1 / weighted-F1 / OOV accuracy, dev & test |
| `confusion_test.png` | Row-normalised confusion matrices, Naive Bayes vs HMM |
| `tuning_dev.png` | Dev sweeps of NB α and HMM transition add-k |

## 5. How to Run

```bash
uv sync
uv run src/main.py --retrain   # tune on dev, train, evaluate, plot, build U24AI026_NLP_LAB-7.pdf (~15 s)
uv run src/main.py             # reuse output/results.json; only re-plot and rebuild the report
```

### Code layout

| File | Purpose |
|---|---|
| `src/data.py` | Reads the tab-separated tagged files |
| `src/naive_bayes.py` | Naive Bayes tagger (five features, Laplace smoothing, greedy previous-tag decoding) |
| `src/hmm.py` | Bigram HMM estimation and forward / backward / forward-backward Viterbi decoders |
| `src/metrics.py` | Per-tag precision, recall, F1, confusion counts |
| `src/train_eval.py` | Dev tuning, training, evaluation → `output/results.json` |
| `src/visualize.py` | Charts → `assets/` |
| `src/report.py` | PDF report → `U24AI026_NLP_LAB-7.pdf` |
