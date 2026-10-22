# Lab 8 — Midsem Questions as Programs, and a Noisy-Channel Spell Corrector

The assignment ([`NLP-Lab-Assignment-8-2026.pdf`](NLP-Lab-Assignment-8-2026.pdf)) has two parts:

1. **Solve the midsem questions with programs.** There is one script per question in [`NLP-Midsem-Questions-Monsoon-2026.pdf`](NLP-Midsem-Questions-Monsoon-2026.pdf). Each prints every intermediate quantity.
2. **Design a spell corrector.** A noisy-channel model uses `count_unigram.txt` (prior) and `count_1edit.txt` (channel), and is evaluated on every misspelling in `spell-errors.txt`.

---

## Part A — Midsem Solutions

| Q | Topic | Program | Answer |
|---|---|---|---|
| 1 | FST for noun singular/plural | `src/q1_fst.py` | 9-state transducer (diagram `assets/q1_fst_diagram.png`, full transition table in the report). `+PL → es` after s, x, z, ch, sh, else `s`. Runs both ways (generation and analysis). |
| 2 | WordPiece, 2 merges | `src/q2_wordpiece.py` | Merge 1: `##s + ##t → ##st` (score 3/(3·3) = 0.333). Merge 2: `l + ##o → lo` (10/(10·10) = 0.1). **`newest → n ##e ##w ##e ##st`** |
| 3 | TF-IDF, closest sentences | `src/q3_tfidf.py` | idf(cat, dog, pet) = log₁₀(4/3) = 0.1249, idf(animal) = log₁₀ 2 = 0.3010. **S1 "cat dog animal" and S3 "dog pet animal"** are closest (cosine 0.8719). |
| 4 | Naive Bayes spam, E8 = (0,0,0) | `src/q4_naive_bayes.py` | Spam: 4/7·(1/4)³ = 1/112 = 0.0089. NotSpam: 3/7·(2/3)³ = 8/63 = 0.1270. **E8 → NotSpam** (also NotSpam with Laplace). |
| 5 | Kneser-Ney, d = 0.5 | `src/q5_kneser_ney.py` | **P_KN(AI \| love) = 0.25 + 0.5·0.25 = 3/8 = 0.375**. **P_KN(NLP \| like) = 0.5 + (1/3)·0.25 = 7/12 = 0.5833** |

All intermediate values (pair scores, TF/IDF tables, per-class factors, continuation counts) are in [`output/midsem_solutions.json`](output/midsem_solutions.json) and in the report.

## Part B — Noisy-Channel Spell Corrector

```
c* = argmax_c P(c) · P(x | c)
P(c)      = count(c) / N                                  (count_unigram.txt, 333,333 words)
P(x | c)  = p_err · Π P(edit_i)                           (count_1edit.txt)
  joint:        P(e) = count(e) / Σ count                 (Norvig)
  conditional:  P(e) = count(e) / count(correct side)     (Kernighan, Church & Gale)
```

- **Edit notation.** `x|y` means "typed x where the correct word has y", with `>` as the word-start marker. This direction was *verified* by aligning spell-errors.txt against count_1edit.txt (Pearson r = 0.88 vs 0.72 for the reverse reading).
- **Candidate search.** Candidates are all words within 1 edit (the assignment's model) or 2 edits (extension). They are found by a prefix-trie search that only extends real-word prefixes, about 8 ms per word.
- **Main setting: error detection given.** Every input is a known misspelling, so the input itself is never returned.

### Results (39,507 distinct pairs)

| Scoring | 1 edit | 2 edits |
|---|---:|---:|
| Prior only | 26.12% | 25.63% |
| Channel only | 20.32% | 26.55% |
| Noisy channel, joint P(edit) | 28.30% | **38.91%** |
| Noisy channel, conditional P(edit) | **29.25%** | 37.11% |
| *Oracle: correct word among candidates* | *35.02%* | *59.39%* |

- **Classic setting** (single-edit, non-word errors, 8,577 pairs): **90.1%** with the conditional channel, 88.5% with the joint channel.
- **The ceiling comes from the data.** Only 36% of the misspellings are one edit from the correct word, and 39% are three or more edits away (phonetic errors such as *negoshation*).
- **Leakage check.** count_1edit.txt was built from these same errors. 5-fold CV with edit counts re-estimated on training folds only changes accuracy by at most 0.43 points (e.g. 2-edit joint: 38.91% → 38.48%).
- **Open setting** (input may be returned unchanged, p_err = 0.05): accuracy drops to 19.9% / 30.3%. That is because 26% of the misspellings (*rainning*, *begining*, …) appear in the web-derived unigram list.

## Charts (`assets/`)

| File | Content |
|---|---|
| `q1_fst_diagram.png` | FST state diagram |
| `q3_tfidf_cosine.png` | Cosine-similarity matrix for Q3 |
| `spell_accuracy.png` | Accuracy by scoring model vs oracle |
| `spell_breakdown.png` | Accuracy by error distance and real-word vs non-word |
| `spell_p_err_sweep.png` | Open-setting accuracy vs p_err |
| `spell_cv_leakage.png` | count_1edit vs held-out channel counts |

## How to Run

```bash
uv sync
uv run src/main.py --retrain   # midsem programs + full spell evaluation (~12 min, uses all cores) + charts + U24AI026_NLP_LAB-8.pdf
uv run src/main.py             # reuse output/spell_results.json; only re-plot and rebuild the report
uv run src/q2_wordpiece.py     # any single midsem solution, with step-by-step printout
```

| File | Purpose |
|---|---|
| `src/q1_fst.py` … `src/q5_kneser_ney.py` | One program per midsem question |
| `src/midsem.py` | Runs all five → `output/midsem_solutions.json` |
| `src/noisy_channel.py` | Language model, channel model, trie-based candidate generation |
| `src/spell_eval.py` | Evaluation grid, breakdowns, p_err sweep, cross-validation → `output/spell_results.json` |
| `src/visualize.py` | Charts → `assets/` |
| `src/report.py` | PDF report → `U24AI026_NLP_LAB-8.pdf` |
