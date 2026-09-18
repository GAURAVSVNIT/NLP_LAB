"""Runs the complete Lab-6 pipeline:
1. Fits the 4 n-gram models on Assignment 4's training split.
2. Evaluates the 5 smoothing techniques across orders 1–4 on dev and test sets.
3. Compares against Assignment 4 (Laplace) and Assignment 5 (Add-k) baselines.
4. Generates publication-quality comparison charts in assets/.
5. Compiles the formatted lab report LAB-6-Report.pdf.
"""
import sys
from pathlib import Path

import report
import train_eval
import visualize

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / 'output' / 'results.json'


def main():
    print("=== NLP Lab 6: Advanced Smoothing Techniques Pipeline ===")
    retrain = '--retrain' in sys.argv or '--force' in sys.argv or not RESULTS_FILE.exists()
    
    if retrain:
        train_eval.main()
    else:
        print(f"Loaded existing evaluation from {RESULTS_FILE}")
        print("(Pass --retrain to re-fit models and re-evaluate on train/dev/test)")

    print("\nGenerating comparison charts...")
    visualize.main()

    print("\nBuilding PDF report (LAB-6-Report.pdf)...")
    report.build()

    print("\nPipeline execution finished successfully!")


if __name__ == '__main__':
    main()
