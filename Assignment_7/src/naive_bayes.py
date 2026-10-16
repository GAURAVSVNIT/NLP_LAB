"""Naive Bayes POS tagger.

Each token is classified independently using five categorical features:
  - len:       'MORE' if len(word) > 4 else 'LESS'   (binary length feature)
  - word:      the actual word
  - prev_word: the previous word (START at sentence start)
  - next_word: the next word (END at sentence end)
  - prev_tag:  the previous tag (START at sentence start)

Training uses gold previous tags. At prediction time the previous tag is the
tag predicted for the previous token (greedy left-to-right decoding).
"""
import math
from collections import Counter, defaultdict

from data import START, END


def length_feature(word):
    return 'MORE' if len(word) > 4 else 'LESS'


def token_features(words, i, prev_tag):
    return {
        'len': length_feature(words[i]),
        'word': words[i],
        'prev_word': words[i - 1] if i > 0 else START,
        'next_word': words[i + 1] if i + 1 < len(words) else END,
        'prev_tag': prev_tag,
    }


class NaiveBayesTagger:
    def __init__(self, alpha=1.0):
        # Laplace smoothing constant for each categorical feature
        self.alpha = alpha

    def fit(self, sentences):
        self.tag_count = Counter()
        self.feat_count = defaultdict(Counter)   # feature -> Counter of (tag, value)
        self.feat_values = defaultdict(set)      # feature -> observed values

        for sent in sentences:
            words = [w for w, _ in sent]
            prev_tag = START
            for i, (_, tag) in enumerate(sent):
                self.tag_count[tag] += 1
                for name, value in token_features(words, i, prev_tag).items():
                    self.feat_count[name][(tag, value)] += 1
                    self.feat_values[name].add(value)
                prev_tag = tag

        total = sum(self.tag_count.values())
        self.tags = sorted(self.tag_count)
        self.log_prior = {t: math.log(c / total) for t, c in self.tag_count.items()}
        return self

    def log_score(self, tag, feats):
        score = self.log_prior[tag]
        for name, value in feats.items():
            # +1 in the vocabulary size reserves mass for values unseen in training
            vocab = len(self.feat_values[name]) + 1
            num = self.feat_count[name].get((tag, value), 0) + self.alpha
            den = self.tag_count[tag] + self.alpha * vocab
            score += math.log(num / den)
        return score

    def predict(self, words):
        tags, prev_tag = [], START
        for i in range(len(words)):
            feats = token_features(words, i, prev_tag)
            prev_tag = max(self.tags, key=lambda t: self.log_score(t, feats))
            tags.append(prev_tag)
        return tags
