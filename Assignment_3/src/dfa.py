"""DFA recognizing simplified English words: a lowercase letter followed by
zero or more lowercase letters (Lab 3, Q1)."""
import string

START, ACCEPT, TRAP = 'q0', 'q1', 'trap'
ALPHABET = set(string.ascii_lowercase)


def build_transition_table():
    """Explicit transition table over the full input alphabet. q0 and q1
    both advance to q1 on any lowercase letter; a trap state absorbs
    everything else (digits, punctuation, uppercase, spaces) so the
    automaton stays total, as a DFA requires."""
    table = {state: {ch: ACCEPT for ch in ALPHABET} for state in (START, ACCEPT)}
    table[TRAP] = {ch: TRAP for ch in ALPHABET}
    return table


TRANSITIONS = build_transition_table()
ACCEPT_STATES = {ACCEPT}


def run(word):
    """Simulate the DFA over `word`; returns True iff it ends in an accept
    state. Any character not in TRANSITIONS[state] (i.e. outside a-z)
    forces a transition to TRAP, same as an explicit trap-state entry for
    the rest of the input alphabet would."""
    if not word:
        return False
    state = START
    for ch in word:
        state = TRANSITIONS.get(state, {}).get(ch, TRAP)
    return state in ACCEPT_STATES


def classify(word):
    return 'Accepted' if run(word) else 'Not Accepted'
