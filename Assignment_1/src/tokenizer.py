import regex

# Gujarati honorifics/titles and common English abbreviations that are
# frequently followed by "." without ending the sentence. Matched against
# the WORD token immediately preceding a single "." (case-insensitive,
# so "Dr"/"DR"/"dr" all match).
ABBREVIATIONS = {
    # Gujarati honorifics / titles
    "ડૉ", "ડો", "શ્રી", "શ્રીમતી", "કુ", "કુમારી", "પ્રો", "સૌ",
    # Common English abbreviations that show up in mixed-script text
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "no", "co", "inc", "ltd", "fig", "approx",
}

EMAIL_RE = r"(?P<EMAIL>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
URL_RE = r"(?P<URL>(?:https?://|www\.)\S+)"
DATE_RE = (
    r"(?P<DATE>"
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|\d{4}-\d{1,2}-\d{1,2}"
    r")"
)
NUMBER_RE = (
    r"(?P<NUMBER>"
    r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?%?"
    r"|\d+(?:\.\d+)?%?"
    r")"
)
ELLIPSIS_RE = r"(?P<ELLIPSIS>\.{2,}|…)"
SENT_END_RE = r"(?P<SENTEND>[।॥]|[!?]+)"
PUNCT_RE = (
    r"(?P<PUNCT>[.,;:'\"“”‘’()\[\]{}<>\-–—/\\|@#$%^&*+=~_`॰])"
)
WORD_RE = r"(?P<WORD>[\p{L}\p{M}]+)"
OTHER_RE = r"(?P<OTHER>\S)"

TOKEN_RE = regex.compile(
    "|".join(
        [EMAIL_RE, URL_RE, DATE_RE, NUMBER_RE, ELLIPSIS_RE, SENT_END_RE, PUNCT_RE, WORD_RE, OTHER_RE]
    )
)

TRAILING_URL_PUNCT = ".,;:!?)”’'\""

SENTENCE_ENDING_TYPES = {"SENTEND", "ELLIPSIS"}
SENTENCE_ENDING_PUNCT_TEXT = {".", "!", "?", "।", "॥"}


class Token:
    __slots__ = ("text", "type")

    def __init__(self, text, type_):
        self.text = text
        self.type = type_

    def __repr__(self):
        return f"Token({self.text!r}, {self.type})"


def word_tokenize(text):
    """Tokenize a string into a flat list of Token objects."""
    tokens = []
    for m in TOKEN_RE.finditer(text):
        kind = m.lastgroup
        value = m.group()

        if kind == "URL":
            trimmed = value.rstrip(TRAILING_URL_PUNCT)
            trailing = value[len(trimmed):]
            value = trimmed
            tokens.append(Token(value, "URL"))
            for ch in trailing:
                tokens.append(Token(ch, "PUNCT"))
            continue

        if kind == "PUNCT" and value in SENTENCE_ENDING_PUNCT_TEXT:
            kind = "SENTEND"

        tokens.append(Token(value, kind))
    return tokens


def _is_abbreviation_period(current, tok):
    """True if `tok` is a lone '.' directly after a listed abbreviation/honorific."""
    if tok.type != "SENTEND" or tok.text != ".":
        return False
    if len(current) < 2:
        return False
    prev = current[-2]
    return prev.type == "WORD" and prev.text.lower() in ABBREVIATIONS


def sentence_tokenize(paragraph):
    """Split a paragraph into a list of sentences, each a list of Token objects."""
    tokens = word_tokenize(paragraph)
    sentences = []
    current = []
    for tok in tokens:
        current.append(tok)
        if tok.type in SENTENCE_ENDING_TYPES:
            if _is_abbreviation_period(current, tok):
                continue
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def tokenize_paragraph(paragraph):
    """Return list[list[str]]: sentences of word-token strings for a paragraph."""
    return [[tok.text for tok in sent] for sent in sentence_tokenize(paragraph)]
