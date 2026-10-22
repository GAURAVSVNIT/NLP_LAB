"""Per-tag precision / recall / F1, accuracy, and confusion counts."""
from collections import Counter


def evaluate(gold, pred, tags, known_mask=None):
    """gold, pred: flat lists of tags. known_mask: flat list of bools (word seen in training)."""
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion = Counter()
    correct = 0
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[g] += 1
            confusion[(g, p)] += 1

    per_tag = {}
    for t in tags:
        prec = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        rec = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_tag[t] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': tp[t] + fn[t]}

    present = [t for t in tags if per_tag[t]['support'] > 0]
    total = sum(per_tag[t]['support'] for t in present)
    result = {
        'accuracy': correct / len(gold),
        'macro_f1': sum(per_tag[t]['f1'] for t in present) / len(present),
        'weighted_f1': sum(per_tag[t]['f1'] * per_tag[t]['support'] for t in present) / total,
        'per_tag': per_tag,
        'top_confusions': [{'gold': g, 'pred': p, 'count': c} for (g, p), c in confusion.most_common(12)],
    }
    if known_mask is not None:
        for name, want in [('known', True), ('oov', False)]:
            pairs = [(g, p) for g, p, k in zip(gold, pred, known_mask) if k == want]
            result[f'{name}_accuracy'] = sum(g == p for g, p in pairs) / len(pairs) if pairs else 0.0
            result[f'{name}_tokens'] = len(pairs)
    return result


def confusion_matrix(gold, pred, tags):
    idx = {t: i for i, t in enumerate(tags)}
    m = [[0] * len(tags) for _ in tags]
    for g, p in zip(gold, pred):
        m[idx[g]][idx[p]] += 1
    return m
