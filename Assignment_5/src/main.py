"""Runs the full Lab-5 pipeline: train the four n-gram models on Assignment
4's split, evaluate Add-k (k=0.3) smoothed perplexity, draw comparison
charts against the Laplace baseline, and build the formatted PDF report.
"""
import report
import train_eval
import visualize


def main():
    train_eval.main()
    visualize.main()
    report.build()


if __name__ == '__main__':
    main()
