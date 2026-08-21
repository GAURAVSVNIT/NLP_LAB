"""Trie (prefix tree) over the vocabulary, used to speed up word matching in the
greedy and DP segmentation algorithms."""
import math


class TrieNode:
    __slots__ = ('children', 'is_word', 'log_prob')

    def __init__(self):
        self.children = {}
        self.is_word = False
        self.log_prob = None


def build_trie(word_counts, total_words):
    """Insert every vocabulary word into a trie, storing each word's log-probability
    (log(count / total_words)) at its terminal node."""
    root = TrieNode()
    for word, count in word_counts.items():
        node = root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True
        node.log_prob = math.log(count / total_words)
    return root
