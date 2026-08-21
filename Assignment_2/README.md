# Lab 2 — Text Segmentation (Brown Corpus)

Segments unspaced text (`"itthatthecity..."`) back into words using two approaches, and
evaluates both against 1000 ground-truth test cases from `text_segmentation_dataset.json`
(a Brown-corpus snapshot: 1500-word vocabulary with frequency counts, 735,040 total corpus
words, 1000 labeled test cases).

## Approach

- **Greedy longest-match** (`src/segmentation.py::greedy_segment`) — at each position, take
  the longest vocabulary word that matches the remaining text; if nothing matches, fall back
  to a single character so progress is always made. Classic maximum-matching segmentation —
  fast, but a single wrong greedy choice early on derails everything after it (e.g. matching
  `"takes"` when the correct split was `"take" + "steps"`).
- **Dynamic programming over log-probability** (`src/segmentation.py::dp_segment`) — finds
  the word sequence that maximizes total log P(word) under a unigram model built from the
  corpus frequencies (`P(w) = count(w) / total_corpus_words`). `best[i]` holds the best score
  for segmenting `text[:i]`; transitions try every word ending at `i` up to the vocabulary's
  max word length, so it's O(n · max_word_len). Substrings outside the 1500-word vocabulary
  get a small length-penalized fallback probability (Norvig-style: `10 / (N · 10^len)`) so
  the DP never gets stuck, though every ground-truth word here is in-vocabulary.
- **Trie-accelerated versions** (`src/trie.py`, `greedy_segment_trie` / `dp_segment_trie`) —
  identical algorithms, reimplemented on a prefix tree of the vocabulary instead of a plain
  dict. Both original versions repeatedly slice out a candidate substring and hash it against
  the vocabulary for every one of the `max_word_len` candidate lengths at each position. A
  trie walk instead advances one character at a time, so it finds every matching word length
  in a single forward pass and stops immediately once no vocabulary word can possibly match —
  no wasted slicing/hashing on dead-end prefixes. `main.py` asserts both trie outputs are
  identical to the dict-based outputs, position for position, and reports timing for all four.

## Metrics (`src/metrics.py`)

- **Accuracy** — exact-match rate: fraction of test cases where the predicted word sequence
  equals the ground truth exactly.
- **Edit distance** — Levenshtein distance between the space-joined predicted and
  ground-truth strings, averaged over the test set (raw and length-normalized).

## Results (1000 test cases, 1500-word vocabulary)

| Metric                        |   Greedy | Greedy (trie) |       DP | DP (trie) |
|--------------------------------|---------:|--------------:|---------:|----------:|
| Exact-match accuracy           |   69.10% |         69.10% |   98.20% |    98.20% |
| Avg edit distance               |    1.290 |          1.290 |    0.028 |     0.028 |
| Runtime, 1000 cases (s)         |    0.012 |          0.006 |    0.169 |     0.129 |

Trie and dict versions score identically (they're the same algorithm) — the trie only changes
*how fast* the matching happens, not *what* it matches. Greedy sees the bigger relative
speedup (~2x) since most of its cost is the repeated slice-and-hash of failed candidates;
DP's speedup is smaller here because the fallback-probability loop (length-only, no trie
involved) still dominates its per-position cost at this vocabulary size.

Full numbers in [`output/results.json`](output/results.json); qualitative examples in
[`output/sample_predictions.json`](output/sample_predictions.json).

**Takeaway:** greedy longest-match is ~15x faster but far less accurate — it has no way to
recover once it commits to a locally-longer word that turns out wrong. The DP approach
looks at word frequency (not just length) and finds the globally best-scoring split, at the
cost of scanning more candidate substrings per position.

## Run it

```bash
uv run --python 3.13 python src/main.py
```
