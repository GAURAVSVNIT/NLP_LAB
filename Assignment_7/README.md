# Lab 7 — Hindi POS Tagging: Naive Bayes and HMM Viterbi Variants

This lab builds part-of-speech taggers for Hindi, trained on the train split and evaluated on the dev and test splits (`Lab-Assignment-7/Hindi-POS-Data`). It follows the same layout as [Assignment 6](../Assignment_6).

1. **Naive Bayes** tagger over five features per token: length (`MORE` if the word is longer than 4 characters, else `LESS`), the word, the previous word, the next word, and the previous tag.
2. **First-order HMM** tagger, decoded three ways:
   - **Forward Viterbi**: standard left-to-right Viterbi.
   - **Backward Viterbi**: right-to-left Viterbi over a reverse-direction transition model `P(t_i | t_{i+1})`.
   - **Forward-backward Viterbi**: a forward and a backward max-product pass; each token gets the tag on the best full-sequence path through it (max-marginal decoding). Its output is per token, so it need not form one consistent tag path.

## Implementation notes

- Naive Bayes uses Laplace smoothing (`alpha = 1`) per feature. Training uses gold previous tags; prediction is greedy, so it uses the predicted previous tag.
- HMM transitions use add-one smoothing. Emissions are MLE. Words seen once in training are mapped to `<UNK>`, so unknown test words get the emission profile of rare words.
- The three Viterbi decoders were checked against brute-force search over all tag sequences on short sentences (forward and backward: exact joint argmax; forward-backward: exact max-marginals).

## Results (token level)

| Method | Dev accuracy | Dev macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.9102 | 0.7436 | 0.9104 | 0.7600 |
| HMM forward Viterbi | 0.9375 | 0.9400 | 0.9372 | 0.9067 |
| HMM backward Viterbi | 0.9372 | 0.9406 | 0.9363 | 0.9055 |
| HMM forward-backward Viterbi | 0.9375 | 0.9400 | 0.9372 | 0.9067 |

Macro-F1 averages the per-tag F1 over tags with gold support. Per-tag F1 for every method is in [`output/results.json`](output/results.json).

## Analysis

- **The HMM beats Naive Bayes** by about 2.7 accuracy points and 15 macro-F1 points on test. Naive Bayes tags each token independently, given only the previous tag, so it misses tag-sequence structure. The HMM uses transitions across the whole sentence.
- **Naive Bayes fails on rare and closed-class tags.** `PR_PRQ` (F1 0.06) and `RP_INTF` (0.34) are close to unlearnable from its features, and `RB` (0.38) is poor too. Naive Bayes gives the largest per-tag gains from sequence modelling, as expected.
- **The three HMM decoders are nearly identical.** Forward and forward-backward Viterbi produce the same test accuracy (0.9372). Backward Viterbi is about 0.1 points lower. The backward model is estimated from the same bigram counts but with a different normalisation, so its optimum differs slightly.
- **Hardest tags for all methods:** `N_NNP` (proper noun, test F1 about 0.85, 5,296 tokens), `RB` (about 0.85), and `RP_INTF` (about 0.87). Proper nouns are often unseen words, and `RB` overlaps with adverbial uses of other tags.
- **`RP_INJ` has F1 = 0 for all methods** because it occurs once in the test set and no decoder gets it right. Its bar is zero-length in the chart.

## Charts

- [`assets/per_tag_f1_test.png`](assets/per_tag_f1_test.png): per-tag F1 on test, all four methods. Tags are ordered by training frequency and labelled with their test support.
- [`assets/per_tag_f1_dev.png`](assets/per_tag_f1_dev.png): the same on dev.
- [`assets/overall_comparison.png`](assets/overall_comparison.png): accuracy and macro-F1 on dev and test.

## How to run

```bash
cd Assignment_7
uv sync            # or: pip install matplotlib numpy
cd src
python main.py            # uses output/results.json if present
python main.py --retrain  # refit and re-evaluate from scratch
```

Run from `src/`, since the modules import each other by bare name (as in Assignment 6).
