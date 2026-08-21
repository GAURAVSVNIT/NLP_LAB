"""Evaluation metrics for text segmentation: exact-match accuracy and edit distance."""


def levenshtein(a, b):
    """O(len(a) * len(b)) edit distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[lb]


def evaluate_segmentation(predictions, ground_truths):
    """predictions/ground_truths: parallel lists of word-lists, one per test case.

    Accuracy is exact-match at the sentence level (predicted word sequence equals
    ground truth). Edit distance is Levenshtein distance between the space-joined
    predicted and ground-truth strings, reported both raw and length-normalized.
    """
    assert len(predictions) == len(ground_truths)
    n = len(predictions)
    exact_matches = 0
    total_edit_distance = 0
    total_normalized = 0.0

    for pred_words, gt_words in zip(predictions, ground_truths):
        pred_str = ' '.join(pred_words)
        gt_str = ' '.join(gt_words)
        if pred_words == gt_words:
            exact_matches += 1
        dist = levenshtein(pred_str, gt_str)
        total_edit_distance += dist
        total_normalized += dist / max(len(gt_str), 1)

    return {
        'exact_match_accuracy': exact_matches / n,
        'avg_edit_distance': total_edit_distance / n,
        'avg_normalized_edit_distance': total_normalized / n,
    }
