"""Run greedy and DP segmentation over the Brown-corpus test set and report metrics."""
import json
import time
from pathlib import Path

from segmentation import (
    build_max_word_len,
    dp_segment,
    dp_segment_trie,
    greedy_segment,
    greedy_segment_trie,
)
from trie import build_trie
from metrics import evaluate_segmentation

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'text_segmentation_dataset.json'
OUTPUT_DIR = ROOT / 'output'


def main():
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)

    word_counts = data['word_counts']
    vocab = set(word_counts.keys())
    total_words = data['metadata']['total_corpus_words']
    test_cases = data['test_cases']
    max_word_len = build_max_word_len(vocab)

    ground_truths = [case['ground_truth'].split() for case in test_cases]
    trie_root = build_trie(word_counts, total_words)

    t0 = time.perf_counter()
    greedy_preds = [greedy_segment(case['input'], vocab, max_word_len) for case in test_cases]
    t1 = time.perf_counter()
    dp_preds = [dp_segment(case['input'], word_counts, total_words, max_word_len) for case in test_cases]
    t2 = time.perf_counter()
    greedy_trie_preds = [greedy_segment_trie(case['input'], trie_root) for case in test_cases]
    t3 = time.perf_counter()
    dp_trie_preds = [dp_segment_trie(case['input'], trie_root, total_words, max_word_len) for case in test_cases]
    t4 = time.perf_counter()

    # the trie-based versions implement the exact same algorithms, so their
    # output must match the dict-based versions exactly, position for position
    assert greedy_preds == greedy_trie_preds, "greedy trie output diverged from dict version"
    assert dp_preds == dp_trie_preds, "DP trie output diverged from dict version"

    greedy_metrics = evaluate_segmentation(greedy_preds, ground_truths)
    dp_metrics = evaluate_segmentation(dp_preds, ground_truths)
    greedy_trie_metrics = evaluate_segmentation(greedy_trie_preds, ground_truths)
    dp_trie_metrics = evaluate_segmentation(dp_trie_preds, ground_truths)

    report = {
        'greedy': {**greedy_metrics, 'runtime_sec': t1 - t0},
        'dp': {**dp_metrics, 'runtime_sec': t2 - t1},
        'greedy_trie': {**greedy_trie_metrics, 'runtime_sec': t3 - t2},
        'dp_trie': {**dp_trie_metrics, 'runtime_sec': t4 - t3},
        'test_case_count': len(test_cases),
        'vocabulary_size': len(vocab),
        'outputs_match_dict_version': True,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    samples = [
        {
            'input': case['input'],
            'ground_truth': case['ground_truth'],
            'greedy': ' '.join(g),
            'dp': ' '.join(d),
        }
        for case, g, d in list(zip(test_cases, greedy_preds, dp_preds))[:15]
    ]
    with open(OUTPUT_DIR / 'sample_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    print(f"Test cases: {len(test_cases)}   Vocabulary size: {len(vocab)}\n")
    print("Trie-based outputs verified identical to dict-based outputs (assertions passed).\n")
    header = f"{'Metric':<32}{'Greedy':>13}{'Greedy(trie)':>15}{'DP':>13}{'DP(trie)':>13}"
    print(header)
    print('-' * len(header))
    print(f"{'Exact-match accuracy':<32}{greedy_metrics['exact_match_accuracy'] * 100:>12.2f}%{greedy_trie_metrics['exact_match_accuracy'] * 100:>14.2f}%{dp_metrics['exact_match_accuracy'] * 100:>12.2f}%{dp_trie_metrics['exact_match_accuracy'] * 100:>12.2f}%")
    print(f"{'Avg edit distance':<32}{greedy_metrics['avg_edit_distance']:>13.3f}{greedy_trie_metrics['avg_edit_distance']:>15.3f}{dp_metrics['avg_edit_distance']:>13.3f}{dp_trie_metrics['avg_edit_distance']:>13.3f}")
    print(f"{'Runtime (s), 1000 cases':<32}{t1 - t0:>13.3f}{t3 - t2:>15.3f}{t2 - t1:>13.3f}{t4 - t3:>13.3f}")


if __name__ == '__main__':
    main()
