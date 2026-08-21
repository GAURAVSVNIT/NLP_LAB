"""FST for English noun morphology (Lab 3, Q2): analyzes a surface noun into
root+category+number, e.g. foxes -> fox+N+PL, fox -> fox+N+SG, or reports
"Invalid Word" when the spelling matches none of the three orthographic
rules below.

    Rule            Surface pattern         Root recovered
    E insertion     s/z/x/ch/sh root + es   strip "es"
    Y replacement   consonant + y -> ies    strip "ies", append "y"
    S addition      root + s                strip "s"

The transducer reads the word right-to-left, one character per transition
(English's morphology lives in the suffix, so scanning from the end means
every decision only ever needs the character just consumed). States:

    Q0 --'s'--> Q_S --'e'--> Q_SE --'i'--> Q_SEI
     \\--else--> COPY_SG        \\--else--> [e-insertion check]
     Q_S--else--> [plain-s check]

Two branch points are structurally ambiguous from spelling alone: a root
ending in a sibilant/ch/sh cluster before "es" can't be told apart from a
root that simply already ends in "e" (e.g. "houses" parses equally well as
hous+es or house+s); likewise "...ies" can't be told apart from a root
that already ends in "ie" (e.g. "movies" as movy+ies or movie+s). Real
two-level morphology resolves this by composing the rule transducer with a
Lexicon FST; here the corpus vocabulary itself stands in as that lexicon.
"""

SIBILANTS = {'s', 'z', 'x'}
VOWELS = {'a', 'e', 'i', 'o', 'u'}

Q0, Q_S, Q_SE, Q_SEI = 'Q0', 'Q_S', 'Q_SE', 'Q_SEI'
COPY_PL, COPY_SG, REJECT = 'COPY_PL', 'COPY_SG', 'REJECT'

PLURAL_TAG = '+N+PL'
SINGULAR_TAG = '+N+SG'
INVALID = 'Invalid Word'


def _prefer(candidate_a, candidate_b, lexicon):
    """Disambiguate two structurally-valid root readings against the
    lexicon oracle; `candidate_a` (the more 'regular' rule reading) wins
    unless only `candidate_b` is a known word."""
    if candidate_b in lexicon and candidate_a not in lexicon:
        return candidate_b
    return candidate_a


def analyze(word, lexicon, trace=None):
    """Run the FST over `word`, consulting `lexicon` (a set of known
    corpus words) at the two ambiguous branch points. If `trace` is a
    list, the sequence of visited states is appended to it."""
    def visit(state):
        if trace is not None:
            trace.append(state)
        return state

    if not word or not word.isalpha() or not word.islower():
        visit(REJECT)
        return INVALID

    n = len(word)

    def char_from_end(k):
        return word[n - 1 - k] if k < n else None

    visit(Q0)
    if char_from_end(0) != 's':
        visit(COPY_SG)
        return word + SINGULAR_TAG
    visit(Q_S)

    c1 = char_from_end(1)
    if c1 != 'e':
        root, last = word[:-1], c1
        if last == 's':
            # a base noun ending in a doubled "ss" (boss, actress, abyss) --
            # regular pluralization never produces "ss" via S addition on an
            # s-ending root (that always needs e-insertion instead: bus ->
            # buses, never buss), so this is the word's own singular form.
            visit(COPY_SG)
            return word + SINGULAR_TAG
        if last in SIBILANTS:
            visit(REJECT)
            return INVALID  # root ends z/x -- needed e-insertion ("...es")
        if last == 'h' and char_from_end(2) in {'c', 's'}:
            visit(REJECT)
            return INVALID  # root ends ch/sh -- needed e-insertion ("...es")
        if last == 'y' and char_from_end(2) not in (VOWELS | {None}):
            visit(REJECT)
            return INVALID  # consonant+y -- needed y-replacement ("...ies")
        visit(COPY_PL)
        return root + PLURAL_TAG
    visit(Q_SE)

    c2 = char_from_end(2)
    if c2 == 'i':
        visit(Q_SEI)
        preceding = char_from_end(3)
        y_root, e_root = word[:-3] + 'y', word[:-1]
        if preceding is None:
            visit(REJECT)
            return INVALID
        chosen = (
            _prefer(e_root, y_root, lexicon) if preceding in VOWELS
            else _prefer(y_root, e_root, lexicon)
        )
        visit(COPY_PL)
        return chosen + PLURAL_TAG

    is_sibilant_cluster = c2 in SIBILANTS or (c2 == 'h' and char_from_end(3) in {'c', 's'})
    visit(COPY_PL)
    if is_sibilant_cluster:
        return _prefer(word[:-2], word[:-1], lexicon) + PLURAL_TAG
    return word[:-1] + PLURAL_TAG  # root simply ends in "e"; plain S addition
