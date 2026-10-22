"""Midsem Q1 — Finite-state transducer for English noun number morphology.

Lexical tape (input)  : stem letters + '+N' + ('+SG' | '+PL')      e.g.  f o x +N +PL
Surface tape (output) : the singular or plural word form            e.g.  foxes

Rule: plural = stem + 's', except stems ending in s, x, z, ch or sh, which take 'es'.

States
    q0  start, nothing read yet
    q1  inside the stem, last letter is "ordinary" (not s, x, z, c, or the h of ch/sh)
    q2  last letter is s                      (sibilant; also the s of a possible 'sh')
    q3  last letter is x or z                 (sibilant)
    q4  last letter is c                      (could become 'ch')
    q5  last two letters are ch or sh         (sibilant)
    q6  read +N after a regular stem          (q1, q4)
    q7  read +N after a sibilant stem         (q2, q3, q5)
    q8  final: number feature emitted
Letters are copied unchanged (l:l). '+N' and '+SG' are rewritten to the empty string; '+PL' is
rewritten to 's' from q6 and to 'es' from q7.
"""
from collections import deque

LETTERS = 'abcdefghijklmnopqrstuvwxyz'
EPS = ''
STATES = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']
STATE_DESC = {
    'q0': 'start', 'q1': 'stem, last letter ordinary', 'q2': "stem ends in 's'",
    'q3': "stem ends in 'x' or 'z'", 'q4': "stem ends in 'c'", 'q5': "stem ends in 'ch' or 'sh'",
    'q6': '+N read, regular stem', 'q7': '+N read, sibilant stem', 'q8': 'final (accept)',
}
START, FINALS = 'q0', {'q8'}
STEM_STATES = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5']
SIBILANT_STEM = {'q2', 'q3', 'q5'}
REGULAR_STEM = {'q1', 'q4'}


def _letter_target(state, letter):
    if letter == 's':
        return 'q2'
    if letter in 'xz':
        return 'q3'
    if letter == 'c':
        return 'q4'
    if letter == 'h' and state in ('q2', 'q4'):
        return 'q5'
    return 'q1'


def build_transitions():
    """(state, input symbol) -> list of (next state, output string)."""
    delta = {}
    for q in STEM_STATES:
        for ch in LETTERS:
            delta[(q, ch)] = [(_letter_target(q, ch), ch)]
    for q in REGULAR_STEM:
        delta[(q, '+N')] = [('q6', EPS)]
    for q in SIBILANT_STEM:
        delta[(q, '+N')] = [('q7', EPS)]
    delta[('q6', '+SG')] = [('q8', EPS)]
    delta[('q6', '+PL')] = [('q8', 's')]
    delta[('q7', '+SG')] = [('q8', EPS)]
    delta[('q7', '+PL')] = [('q8', 'es')]
    return delta


DELTA = build_transitions()


def tokenize_lexical(form):
    """'fox+N+PL' -> ['f', 'o', 'x', '+N', '+PL']"""
    stem, *feats = form.split('+')
    return list(stem) + ['+' + f for f in feats]


def generate(lexical):
    """Runs the FST in the lexical -> surface direction. Returns the surface form or None."""
    states = [(START, '')]
    for sym in tokenize_lexical(lexical):
        states = [(nxt, out + o) for q, out in states for nxt, o in DELTA.get((q, sym), [])]
    results = [out for q, out in states if q in FINALS]
    return results[0] if results else None


def analyse(surface):
    """Runs the inverted FST (surface -> lexical). Returns every accepting analysis."""
    inverse = {}
    for (q, sym), arcs in DELTA.items():
        for nxt, out in arcs:
            inverse.setdefault(q, []).append((out, sym, nxt))
    found, queue, seen = [], deque([(START, 0, '')]), set()
    while queue:
        q, i, lex = queue.popleft()
        if (q, i, lex) in seen:
            continue
        seen.add((q, i, lex))
        if q in FINALS and i == len(surface):
            found.append(lex)
            continue
        for out, sym, nxt in inverse.get(q, []):
            if surface.startswith(out, i):
                queue.append((nxt, i + len(out), lex + sym))
    return sorted(set(found))


SYMBOL_CLASSES = [
    ('s', 's'), ('x', 'x, z'), ('c', 'c'), ('h', 'h'), ('a', 'other letter'),
    ('+N', '+N'), ('+SG', '+SG'), ('+PL', '+PL'),
]


def transition_table():
    """Rows: states; columns: input-symbol classes; cells: 'next state, input:output'."""
    rows = []
    for q in STATES:
        row = {'state': q, 'description': STATE_DESC[q]}
        for sym, label in SYMBOL_CLASSES:
            arcs = DELTA.get((q, sym))
            if not arcs:
                row[label] = '—'
                continue
            nxt, out = arcs[0]
            if sym.startswith('+'):
                row[label] = f'{nxt}, {sym}:{out or "ε"}'
            else:
                letter = 'l' if label == 'other letter' else label.replace(', ', '/')
                row[label] = f'{nxt}, {letter}:{letter}'
        rows.append(row)
    return rows


TEST_NOUNS = ['cat', 'dog', 'book', 'fox', 'box', 'bus', 'glass', 'buzz', 'quiz', 'church', 'watch',
              'bench', 'dish', 'brush', 'wish', 'month', 'stomach', 'epoch', 'zinc', 'path', 'bath']


def solve():
    table = transition_table()
    generated = [{'noun': n, 'singular': generate(f'{n}+N+SG'), 'plural': generate(f'{n}+N+PL')}
                 for n in TEST_NOUNS]
    analyses = {w: analyse(w) for w in ['foxes', 'cats', 'churches', 'dishes', 'fox', 'buses', 'months', 'stomachs']}
    return {
        'states': STATE_DESC,
        'start': START,
        'finals': sorted(FINALS),
        'transition_table': table,
        'generated': generated,
        'analyses': analyses,
    }


if __name__ == '__main__':
    out = solve()
    for r in out['transition_table']:
        print(r)
    for g in out['generated']:
        print(f"{g['noun']:10s} +SG -> {g['singular']:10s} +PL -> {g['plural']}")
    for w, a in out['analyses'].items():
        print(f'{w:10s} -> {a}')
