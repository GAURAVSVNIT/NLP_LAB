# Lab 3 — DFA for Word Recognition & FST for Noun Morphology

Two finite-state machines over the Brown corpus's noun list (`brown_nouns.txt`,
202,793 tokens / 19,287 unique word types):

1. A **DFA** (`src/dfa.py`) that recognizes "simplified English words" — a lowercase
   letter followed by zero or more lowercase letters.
2. An **FST** (`src/fst.py`) that analyzes a surface noun into `root+N+SG` or
   `root+N+PL` (e.g. `foxes -> fox+N+PL`), or reports `Invalid Word` when the
   spelling doesn't match any of the three orthographic rules.

## Q1 — DFA (`src/dfa.py`)

States `{q0, q1, trap}`, alphabet `a`–`z` (plus an implicit trap for everything
else — digits, punctuation, uppercase, spaces), accept state `{q1}`:

| State | on a–z | on anything else |
|-------|--------|-------------------|
| q0 (start)   | → q1 | → trap |
| q1 (accept)  | → q1 | → trap |
| trap         | → trap | → trap |

The empty string is rejected (a word must *start* with a letter), and the trap
state makes the transition function total, as a DFA requires — a digit,
underscore, capital, or leading space anywhere sends the automaton to trap for
good. Verified against the assignment's own examples (`cat`, `dog`, `a`,
`zebra` accepted; `dog1`, `1dog`, `DogHouse`, `Dog_house`, a leading-space
`" cats"` rejected), then run over all 19,287 corpus word types as a sanity
check: 17,053 accepted, 2,234 rejected — the rejects are exactly the
non-word tokens the Brown noun tagger let through (currency figures, hyphen-
joined compounds, etc., e.g. `$.027`, `$1,000`).

## Q2 — FST (`src/fst.py`)

| Rule | Surface pattern | Root recovered |
|------|------------------|-----------------|
| E insertion   | s/z/x/ch/sh root + `es` | strip `es` |
| Y replacement | consonant + y → `ies`   | strip `ies`, append `y` |
| S addition    | root + `s`              | strip `s` |

The transducer reads the word **right-to-left**, one character per
transition — English morphology lives in the suffix, so scanning from the end
means every decision only ever needs the character just consumed:

```
Q0 --'s'--> Q_S --'e'--> Q_SE --'i'--> Q_SEI --[valid?]--> COPY_PL
 \--else--> COPY_SG         \--else--> [e-insertion check]
  \--[plain-s check]--> COPY_PL | REJECT
```

`COPY_SG`/`COPY_PL`/`REJECT` are the accepting (or rejecting) sinks that
determine the output tag; `src/fst.py` implements this as a literal
state-by-state walk (see `analyze()`), and `output/fst_sample_predictions.json`
records the exact `states_visited` list for each example so the walk is
inspectable.

**Ambiguity and the lexicon oracle.** Two branch points can't be resolved from
spelling alone:

- `...es`: a root ending in a sibilant/`ch`/`sh` cluster (`fox+es`) is
  spelled identically to a root that already ends in `e` (`house+s = houses`).
- `...ies`: a y-replaced root (`try -> tries`) is spelled identically to a
  root that already ends in `ie` (`movie -> movies`).

Real two-level morphology resolves this by composing the rule transducer with
a **Lexicon FST** that only accepts genuine dictionary roots. Here the corpus
vocabulary itself stands in as that lexicon: at both branch points, the FST
prefers whichever candidate root (e.g. `hous` vs. `house`, or `movy` vs.
`movie`) is an attested word elsewhere in the corpus, falling back to the
"more regular" rule reading when neither (or both) are attested. This
correctly resolves `houses -> house+N+PL` (not `hous+N+PL`) and
`movies -> movie+N+PL` (not `movy+N+PL`).

A bare-`s` ending also gets one non-ambiguous correction: a word ending in a
doubled `ss` (`boss`, `actress`, `abyss`) is always treated as its own
singular root, never as `root-ending-in-s` + plain `s` — regular
pluralization of an s-ending root always goes through e-insertion instead
(`bus -> buses`, never `buss`), so a bare `...ss` can only be the word's own
spelling.

**Invalid words.** Anything that isn't lowercase a–z fails immediately
(reusing the DFA's word-shape check), and any surface form that skips a
required rule is rejected, e.g. `foxs` (should be `foxes`, e-insertion
skipped) or `trys` (should be `tries`, y-replacement skipped).

## Results (17,053 DFA-accepted corpus word types)

| Metric | Count |
|--------|------:|
| Singular (`+N+SG`) | 11,113 |
| Plural (`+N+PL`)   | 5,937 |
| Invalid            | 3 |

The 3 residual "invalid" words are genuine exceptions outside the three
named rules, not bugs: `stomachs` (the `ch` here is pronounced /k/, not the
affricate the e-insertion rule assumes — same class as `chemist`, `monarch`),
and the loanword/proper-noun plurals `zlotys`, `warys`, which don't undergo
native y-replacement. Full per-word output in
[`output/fst_results.json`](output/fst_results.json); example traces and the
summary in [`output/fst_sample_predictions.json`](output/fst_sample_predictions.json).
DFA example/corpus results in [`output/dfa_results.json`](output/dfa_results.json).

**Note on scope:** the FST only implements the three named rules, so
irregular plurals (`children`, `men`, `mice`) don't end in `s` and are
reported as `+N+SG` — expected given the assignment's rule set, not a defect.

## Run it

```bash
uv run --python 3.13 python src/main.py
```
