"""Evaluates the noisy-channel spell corrector on every (correct, misspelling) pair in spell-errors.txt.

Settings
    detected  the input is known to be misspelled (every entry of spell-errors.txt is), so the typo
              itself is never returned. This is the classical non-word correction set-up of
              Kernighan et al. (1990) and the main setting of this lab.
    open      the typo is also a candidate, with P(x|x) = 1 - p_err. count_unigram.txt is built from
              web text, so 26% of the misspellings (e.g. 'rainning', 'begining') are themselves in
              the vocabulary. This setting shows how much that noise costs.

Scoring modes: prior only, channel only, noisy channel (joint P(edit)), noisy channel
(conditional P(edit)). Candidates are all words within edit distance 1 (the assignment's model) or
2 (extension).

Data leakage check: count_1edit.txt was derived from this very error list. Every edit we can align in
spell-errors.txt appears in it. A 5-fold cross-validation, grouped by correct word, re-estimates the
edit counts from the four training folds only and evaluates on the held-out fold.
"""
import json
import multiprocessing as mp
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from noisy_channel import (
    DATA_DIR, NoisyChannelSpeller, build_prefixes, candidate_edits, load_edits, load_unigrams
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

MODES = ['prior', 'channel', 'noisy_joint', 'noisy_conditional']
P_ERR_DEFAULT = 0.05
P_ERR_SWEEP = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
N_FOLDS = 5
SEED = 7

_WORDS = _PREFIXES = None


def load_pairs(path=DATA_DIR / 'spell-errors.txt'):
    """Returns (pairs, n_identical). pairs: list of dicts with correct, typo, weight (token count), line."""
    agg = Counter()
    line_of = {}
    identical = 0
    for ln, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        right, wrongs = line.split(':', 1)
        right = right.strip().lower().replace('_', ' ')
        for w in wrongs.split(','):
            w = w.strip()
            m = re.match(r'(.*)\*(\d+)$', w)
            n = int(m.group(2)) if m else 1
            w = (m.group(1) if m else w).lower().replace('_', ' ')
            if w == right:
                identical += n
                continue
            agg[(right, w)] += n
            line_of.setdefault((right, w), ln)
    pairs = [{'correct': r, 'typo': t, 'weight': n, 'line': line_of[(r, t)]} for (r, t), n in agg.items()]
    return pairs, identical


def damerau_levenshtein(a, b):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _init_worker(words, prefixes):
    global _WORDS, _PREFIXES
    _WORDS, _PREFIXES = words, prefixes


def _cands_worker(typo):
    return typo, candidate_edits(typo, _WORDS, _PREFIXES, 2)


def all_candidates(typos, words, prefixes):
    ctx = mp.get_context('fork')
    with ctx.Pool(initializer=_init_worker, initargs=(words, prefixes)) as pool:
        return dict(pool.imap_unordered(_cands_worker, typos, chunksize=64))


def restrict(cands, max_dist, exclude=None):
    """Keeps candidates reachable within max_dist edits (and only those edit sequences)."""
    out = {}
    for w, edit_sets in cands.items():
        if w == exclude:
            continue
        kept = {e for e in edit_sets if len(e) <= max_dist}
        if kept:
            out[w] = kept
    return out


_RESTRICTED = {}


def restricted(cand_map, typo, max_dist, setting):
    key = (typo, max_dist, setting)
    if key not in _RESTRICTED:
        _RESTRICTED[key] = restrict(cand_map[typo], max_dist, exclude=typo if setting == 'detected' else None)
    return _RESTRICTED[key]


def predict_all(speller, cand_map, typos, max_dist, setting, mode, top_k=1):
    preds = {}
    for t in typos:
        c = restricted(cand_map, t, max_dist, setting)
        preds[t] = speller.rank(c, 'noisy_channel' if mode.startswith('noisy') else mode)[:top_k] if c else []
    return preds


def score(pairs, preds, k=1):
    hit = [p['correct'] in preds[p['typo']][:k] for p in pairs]
    n_tok = sum(p['weight'] for p in pairs)
    return {
        'accuracy': sum(hit) / len(pairs),
        'token_accuracy': sum(p['weight'] for p, h in zip(pairs, hit) if h) / n_tok,
        'n_pairs': len(pairs),
    }


def align_edits(correct, typo):
    """Minimal edit sequence (<= 2 edits) turning `correct` into `typo`, in count_1edit notation."""
    found = candidate_edits(typo, {correct}, build_prefixes([correct]), 2).get(correct)
    if not found:
        return None
    return min(found, key=lambda e: (len(e), e))


def verify_edit_direction(pairs, edit_counts):
    """Aligns every single-insertion/deletion pair and correlates the derived counts with count_1edit
    under both readings of 'a|b'. The reading with the higher correlation is the file's convention."""
    derived = Counter()
    for p in pairs:
        if p['distance'] == 1:
            e = align_edits(p['correct'], p['typo'])
            if e and len(e) == 1:
                derived[e[0]] += p['weight']
    out = {}
    for name, flip in [('typo|correct', False), ('correct|typo', True)]:
        d = Counter({('|'.join(k.split('|')[::-1]) if flip else k): v for k, v in derived.items()})
        keys = [k for k in d if len(k.split('|')[0]) != len(k.split('|')[1])]   # asymmetric indel edits
        xs = [d[k] for k in keys]
        ys = [edit_counts.get(k, 0) for k in keys]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        r = cov / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        out[name] = {'pearson_r': r, 'indel_keys': len(keys), 'missing_from_count_1edit': sum(y == 0 for y in ys)}
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)   # show progress even when output is redirected
    t0 = time.perf_counter()
    unigrams, edit_counts = load_unigrams(), load_edits()
    words = set(unigrams)
    prefixes = build_prefixes(words)
    pairs, identical = load_pairs()
    for p in pairs:
        p['distance'] = damerau_levenshtein(p['correct'], p['typo'])
        p['typo_in_vocab'] = p['typo'] in words
        p['correct_in_vocab'] = p['correct'] in words
    typos = sorted({p['typo'] for p in pairs})
    direction = verify_edit_direction(pairs, edit_counts)
    print(f'{len(pairs):,} unique pairs ({sum(p["weight"] for p in pairs):,} tokens), {len(typos):,} unique typos')

    print('Generating candidates (edit distance <= 2)...')
    cand_map = all_candidates(typos, words, prefixes)
    print(f'  done in {time.perf_counter() - t0:.0f}s')

    spellers = {
        'joint': NoisyChannelSpeller(unigrams, edit_counts, P_ERR_DEFAULT, 'joint'),
        'conditional': NoisyChannelSpeller(unigrams, edit_counts, P_ERR_DEFAULT, 'conditional'),
    }

    def speller_for(mode):
        return spellers['conditional' if mode == 'noisy_conditional' else 'joint']

    results = {
        'data': {
            'unique_pairs': len(pairs), 'tokens': sum(p['weight'] for p in pairs),
            'identical_after_lowercasing': identical, 'unique_typos': len(typos),
            'vocab_size': len(words), 'edit_types': len(edit_counts), 'edit_total': sum(edit_counts.values()),
            'distance_histogram': dict(sorted(Counter(min(p['distance'], 4) for p in pairs).items())),
            'typo_in_vocab_rate': sum(p['typo_in_vocab'] for p in pairs) / len(pairs),
            'correct_in_vocab_rate': sum(p['correct_in_vocab'] for p in pairs) / len(pairs),
        },
        'edit_direction_check': direction,
        'p_err_default': P_ERR_DEFAULT,
        'oracle': {}, 'main': {}, 'breakdown': {}, 'top_k': {}, 'p_err_sweep': {}, 'cross_validation': {},
    }

    # Oracle: is the correct word among the candidates at all?
    for d in [1, 2]:
        for p in pairs:
            p[f'reachable_d{d}'] = p['correct'] in restricted(cand_map, p['typo'], d, 'detected')
        results['oracle'][f'd{d}'] = sum(p[f'reachable_d{d}'] for p in pairs) / len(pairs)

    # Main grid.
    main_preds = {}
    for setting in ['detected', 'open']:
        for d in [1, 2]:
            for mode in MODES:
                preds = predict_all(speller_for(mode), cand_map, typos, d, setting, mode, top_k=5)
                key = f'{setting}|d{d}|{mode}'
                main_preds[key] = preds
                results['main'][key] = score(pairs, preds)
                print(f'  {key:32s} acc={results["main"][key]["accuracy"]:.4f}')

    # Breakdown + top-k for the headline configurations.
    for key in ['detected|d1|noisy_joint', 'detected|d2|noisy_joint', 'detected|d1|noisy_conditional',
                'detected|d2|noisy_conditional', 'open|d1|noisy_joint']:
        preds = main_preds[key]
        groups = defaultdict(list)
        for p in pairs:
            groups[f'distance={min(p["distance"], 3)}{"+" if p["distance"] >= 3 else ""}'].append(p)
            groups['typo is a vocabulary word' if p['typo_in_vocab'] else 'typo is a non-word'].append(p)
            if p['distance'] == 1 and not p['typo_in_vocab']:
                groups['distance=1 non-word (classic setting)'].append(p)
            d = int(key.split('|')[1][1])
            if p[f'reachable_d{d}']:
                groups['correct word reachable'].append(p)
        results['breakdown'][key] = {g: score(ps, preds) for g, ps in sorted(groups.items())}
        results['top_k'][key] = {f'top{k}': score(pairs, preds, k)['accuracy'] for k in [1, 3, 5]}

    # p_err sweep in the open setting (in the detected setting p_err is a constant factor).
    for d in [1, 2]:
        for ch in ['joint', 'conditional']:
            row = {}
            for pe in P_ERR_SWEEP:
                sp = NoisyChannelSpeller(unigrams, edit_counts, pe, ch)
                row[str(pe)] = score(pairs, predict_all(sp, cand_map, typos, d, 'open', 'noisy'))['accuracy']
            results['p_err_sweep'][f'd{d}|{ch}'] = row

    # 5-fold cross-validation with held-out edit counts, grouped by correct word.
    print('Cross-validation with held-out channel counts...')
    lines = sorted({p['line'] for p in pairs})
    random.Random(SEED).shuffle(lines)
    fold_of = {ln: i % N_FOLDS for i, ln in enumerate(lines)}
    alignment = {}
    for p in pairs:
        if p['distance'] <= 2:
            alignment[(p['correct'], p['typo'])] = align_edits(p['correct'], p['typo'])
    cv = defaultdict(list)
    for fold in range(N_FOLDS):
        test_pairs = [p for p in pairs if fold_of[p['line']] == fold]
        held = Counter()
        for p in pairs:
            if fold_of[p['line']] != fold and alignment.get((p['correct'], p['typo'])):
                for e in alignment[(p['correct'], p['typo'])]:
                    held[e] += p['weight']
        fold_typos = sorted({p['typo'] for p in test_pairs})
        for ch in ['joint', 'conditional']:
            for source, counts in [('count_1edit', edit_counts), ('held_out', dict(held))]:
                sp = NoisyChannelSpeller(unigrams, counts, P_ERR_DEFAULT, ch)
                for d in [1, 2]:
                    preds = predict_all(sp, cand_map, fold_typos, d, 'detected', 'noisy')
                    cv[f'{source}|d{d}|{ch}'].append(score(test_pairs, preds)['accuracy'])
    results['cross_validation'] = {
        k: {'mean': sum(v) / len(v), 'folds': v} for k, v in sorted(cv.items())
    }
    results['cross_validation_note'] = (
        f'{N_FOLDS} folds grouped by correct word; held_out counts are re-derived from the training '
        'folds by aligning each pair with <= 2 edits.')

    # Example corrections.
    head = main_preds['detected|d1|noisy_joint']
    head2 = main_preds['detected|d2|noisy_joint']
    rng = random.Random(SEED)
    sample = rng.sample(pairs, 400)
    examples = []
    for want_ok in [True, False]:
        for p in sample:
            got = head[p['typo']][0] if head[p['typo']] else None
            if (got == p['correct']) == want_ok and len([e for e in examples if e['correct_d1'] == want_ok]) < 12:
                examples.append({'typo': p['typo'], 'correct': p['correct'], 'distance': p['distance'],
                                 'pred_d1': got, 'pred_d2': head2[p['typo']][0] if head2[p['typo']] else None,
                                 'top3_d1': head[p['typo']][:3], 'correct_d1': want_ok})
    results['examples'] = examples
    results['assignment_example'] = {
        t: {'d1': main_preds['detected|d1|noisy_joint'][t][:3]} for t in ['rainning', 'raning'] if t in head
    }
    results['seconds'] = time.perf_counter() - t0

    (OUTPUT_DIR / 'spell_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_DIR / "spell_results.json"} ({results["seconds"]:.0f}s)')


if __name__ == '__main__':
    main()
