"""Noisy-channel spelling corrector.

    c* = argmax_{c in candidates(x)}  P(c) * P(x | c)

Language model (prior)  P(c) = count(c) / N, from count_unigram.txt (333,333 lowercase words).

Channel model  P(x | c)
    c == x (no error)       1 - p_err
    c reached by edits e_i  p_err * Π_i P(e_i)

An edit is written as 'typo_side|correct_side', with '>' marking the start of the word, exactly as
in count_1edit.txt (verified against spell-errors.txt):
    substitution  'e|i'   typed e for i
    deletion      't|te'  the e after t was dropped
    insertion     'ea|e'  an extra a was typed after e
    transposition 'ei|ie'
Two estimators for P(e) are implemented:
    'joint'        P(e) = count(e) / Σ count            (Norvig)
    'conditional'  P(e) = count(e) / count(correct_side)  (Kernighan, Church & Gale 1990), where
                   count(correct_side) is the frequency of that 1-2 character string in the
                   frequency-weighted unigram corpus.
Unseen edits receive a pseudo-count of 0.5.

Candidates are all vocabulary words within `max_dist` Damerau-Levenshtein edits. They are found by a
search over a trie of vocabulary prefixes (Norvig, "Natural Language Corpus Data", 2009). The search
only ever extends a prefix of a real word, which makes distance 2 tractable on a 333k-word vocabulary.
"""
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'Spell-Correction'
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
START = '>'
UNSEEN_EDIT_COUNT = 0.5


def load_unigrams(path=DATA_DIR / 'count_unigram.txt'):
    counts = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        w, c = line.split('\t')
        counts[w] = int(c)
    return counts


def load_edits(path=DATA_DIR / 'count_1edit.txt'):
    counts = {}
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        e, c = line.split('\t')
        counts[e] = int(c)
    return counts


def build_prefixes(words):
    prefixes = set()
    for w in words:
        for i in range(len(w) + 1):
            prefixes.add(w[:i])
    return prefixes


def candidate_edits(typo, words, prefixes, max_dist):
    """Returns {candidate: set of edit tuples} for every word within max_dist edits of `typo`."""
    results = {}

    def search(hd, tl, d, edits):
        cand = hd + tl
        if cand in words:
            results.setdefault(cand, set()).add(tuple(edits))
        if d <= 0:
            # Only exact copying of the rest is possible; the check above already covered it.
            return
        extensions = [hd + c for c in ALPHABET if hd + c in prefixes]
        p = hd[-1] if hd else START
        for h in extensions:                                   # correct word has an extra letter
            search(h, tl, d - 1, edits + [f'{p}|{p}{h[-1]}'])
        if not tl:
            return
        search(hd, tl[1:], d - 1, edits + [f'{p}{tl[0]}|{p}'])  # typo has an extra letter
        for h in extensions:
            if h[-1] == tl[0]:
                search(h, tl[1:], d, edits)                    # match
            else:
                search(h, tl[1:], d - 1, edits + [f'{tl[0]}|{h[-1]}'])   # substitution
        if len(tl) >= 2 and tl[0] != tl[1] and hd + tl[1] in prefixes:   # transposition
            search(hd + tl[1], tl[0] + tl[2:], d - 1, edits + [f'{tl[0]}{tl[1]}|{tl[1]}{tl[0]}'])

    search('', typo, max_dist, [])
    return results


class NoisyChannelSpeller:
    def __init__(self, unigrams, edit_counts, p_err=0.05, channel='joint'):
        self.unigrams = unigrams
        self.total = sum(unigrams.values())
        self.log_prior = {w: math.log(c / self.total) for w, c in unigrams.items()}
        self.p_err = p_err
        self.channel = channel
        self.set_edit_counts(edit_counts)

    def set_edit_counts(self, edit_counts):
        self.edit_counts = edit_counts
        self.edit_total = sum(edit_counts.values())
        self._edit_cache = {}
        if self.channel == 'conditional' and not hasattr(self, 'context_counts'):
            self.context_counts = self._context_counts()

    def _context_counts(self):
        """Frequency of every 1- and 2-character string (with '>' start marker) in the weighted corpus."""
        ctx = Counter()
        for w, c in self.unigrams.items():
            s = START + w
            for i in range(len(s)):
                ctx[s[i]] += c
                if i + 1 < len(s):
                    ctx[s[i:i + 2]] += c
        return ctx

    def log_p_edit(self, edit):
        cached = self._edit_cache.get(edit)
        if cached is not None:
            return cached
        count = self.edit_counts.get(edit, UNSEEN_EDIT_COUNT)
        if self.channel == 'joint':
            value = math.log(count / self.edit_total)
        else:
            correct_side = edit.split('|', 1)[1]
            value = math.log(count / max(self.context_counts.get(correct_side, 0), 1))
        self._edit_cache[edit] = value
        return value

    def log_channel(self, edit_sets):
        """log P(x | c), using the most probable alignment when several edit sequences reach c."""
        best = -math.inf
        for edits in edit_sets:
            if not edits:
                return math.log(1 - self.p_err)
            best = max(best, math.log(self.p_err) + sum(self.log_p_edit(e) for e in edits))
        return best

    def rank(self, cands, mode='noisy_channel'):
        """cands: {word: set of edit tuples}. Returns words sorted best-first under the chosen scoring."""
        def score(w):
            if mode == 'prior':
                return self.log_prior[w]
            if mode == 'channel':
                return self.log_channel(cands[w])
            return self.log_prior[w] + self.log_channel(cands[w])
        return sorted(cands, key=score, reverse=True)

    def correct(self, cands, mode='noisy_channel', fallback=None):
        if not cands:
            return fallback
        return self.rank(cands, mode)[0]
