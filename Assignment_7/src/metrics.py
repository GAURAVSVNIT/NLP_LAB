"""Token-level per-tag precision, recall, F1 and overall accuracy / macro-F1."""
from collections import Counter


def evaluate(gold, pred):
    assert len(gold) == len(pred)
    tags = sorted(set(gold) | set(pred))
    tp, fp, fn = Counter(), Counter(), Counter()
    support = Counter(gold)
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    per_tag = {}
    for t in tags:
        precision = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        recall = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_tag[t] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support[t]}

    # Macro-F1 over tags that occur in the gold data
    gold_tags = [t for t in tags if support[t] > 0]
    accuracy = sum(tp.values()) / len(gold)
    macro_f1 = sum(per_tag[t]['f1'] for t in gold_tags) / len(gold_tags)
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'tokens': len(gold), 'per_tag': per_tag}
