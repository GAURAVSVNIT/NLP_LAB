"""Text segmentation algorithms: greedy longest-match and DP log-probability maximization."""
import math

from trie import build_trie


def build_max_word_len(vocab):
    return max(len(w) for w in vocab)


def greedy_segment(text, vocab, max_word_len=None):
    """Greedy longest-match segmentation: repeatedly take the longest vocabulary word
    matching the remaining text; falls back to a single character when nothing matches."""
    if max_word_len is None:
        max_word_len = build_max_word_len(vocab)
    words = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for length in range(min(max_word_len, n - i), 0, -1):
            candidate = text[i:i + length]
            if candidate in vocab:
                matched = candidate
                break
        if matched is None:
            matched = text[i]
        words.append(matched)
        i += len(matched)
    return words


def dp_segment(text, word_counts, total_words, max_word_len=None):
    """DP segmentation that maximizes total log-probability of the resulting word
    sequence under a unigram model built from word_counts. Unseen substrings get a
    length-penalized fallback probability (Norvig-style) so the DP always has a
    valid solution even outside the known vocabulary."""
    if max_word_len is None:
        max_word_len = build_max_word_len(word_counts)

    n = len(text)

    def log_prob(word):
        count = word_counts.get(word)
        if count is not None:
            return math.log(count / total_words)
        return math.log(10.0 / (total_words * 10 ** len(word)))

    best = [float('-inf')] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0

    for i in range(1, n + 1):
        start = max(0, i - max_word_len)
        for j in range(start, i):
            score = best[j] + log_prob(text[j:i])
            if score > best[i]:
                best[i] = score
                back[i] = j

    words = []
    i = n
    while i > 0:
        j = back[i]
        words.append(text[j:i])
        i = j
    words.reverse()
    return words


def greedy_segment_trie(text, root):
    """Same longest-match strategy as greedy_segment, but walks the trie once per
    starting position instead of re-slicing and re-hashing every candidate length."""
    words = []
    i = 0
    n = len(text)
    while i < n:
        node = root
        last_end = -1
        j = i
        while j < n and text[j] in node.children:
            node = node.children[text[j]]
            j += 1
            if node.is_word:
                last_end = j
        if last_end == -1:
            words.append(text[i])
            i += 1
        else:
            words.append(text[i:last_end])
            i = last_end
    return words


def dp_segment_trie(text, root, total_words, max_word_len):
    """Same objective as dp_segment (maximize total log-probability), but for each
    start position j a single trie walk finds every real-word match ending at
    j+1 .. j+max_word_len in one pass, instead of hashing a fresh substring for
    each candidate length. Unknown-substring fallback scores depend only on
    length, so they're computed directly with no text lookup needed, and are
    skipped for any length the trie walk already matched as a real word."""
    n = len(text)
    best = [float('-inf')] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0

    for j in range(n):
        if best[j] == float('-inf'):
            continue

        matched_lengths = set()
        node = root
        k = j
        while k < n and text[k] in node.children:
            node = node.children[text[k]]
            k += 1
            if node.is_word:
                matched_lengths.add(k - j)
                score = best[j] + node.log_prob
                if score > best[k]:
                    best[k] = score
                    back[k] = j

        for length in range(1, min(max_word_len, n - j) + 1):
            if length in matched_lengths:
                continue
            i = j + length
            score = best[j] + math.log(10.0 / (total_words * 10 ** length))
            if score > best[i]:
                best[i] = score
                back[i] = j

    words = []
    i = n
    while i > 0:
        j = back[i]
        words.append(text[j:i])
        i = j
    words.reverse()
    return words
