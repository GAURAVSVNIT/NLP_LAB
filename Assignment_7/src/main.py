"""Runs the complete Lab-7 pipeline:
1. Fits Naive Bayes and the HMM on the Hindi train split.
2. Evaluates four decoders on dev and test (per-tag F1, accuracy, macro-F1).
3. Writes output/results.json and comparison charts in assets/.
"""
import sys

import train_eval
import visualize
from data import OUTPUT_DIR

RESULTS_FILE = OUTPUT_DIR / 'results.json'


def main():
    print("=== NLP Lab 7: Hindi POS Tagging ===")
    if '--retrain' in sys.argv or not RESULTS_FILE.exists():
        train_eval.main()
    else:
        print(f"Loaded existing evaluation from {RESULTS_FILE}")
        print("(Pass --retrain to re-fit models and re-evaluate)")

    print("\nGenerating comparison charts...")
    visualize.main()
    print("\nPipeline execution finished successfully!")


if __name__ == '__main__':
    main()
