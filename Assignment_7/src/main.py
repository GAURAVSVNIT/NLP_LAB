"""Runs the complete Lab-7 pipeline:
1. Tunes smoothing on dev, trains Naive Bayes and the HMM on the training split.
2. Evaluates Naive Bayes and the three HMM decoders (forward, backward, forward-backward Viterbi)
   on dev and test, reporting per-tag precision / recall / F1.
3. Generates comparison charts in assets/.
4. Compiles the formatted lab report U24AI026_NLP_LAB-7.pdf.
"""
import sys
from pathlib import Path

import report
import train_eval
import visualize

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / 'output' / 'results.json'


def main():
    print('=== NLP Lab 7: Hindi POS Tagging Pipeline ===')
    if '--retrain' in sys.argv or not RESULTS_FILE.exists():
        train_eval.main()
    else:
        print(f'Loaded existing evaluation from {RESULTS_FILE}')
        print('(Pass --retrain to re-fit the models and re-evaluate on dev/test)')

    print('\nGenerating comparison charts...')
    visualize.main()

    print('\nBuilding PDF report (U24AI026_NLP_LAB-7.pdf)...')
    report.build()
    print('\nPipeline execution finished successfully!')


if __name__ == '__main__':
    main()
