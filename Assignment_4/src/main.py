"""Runs the full Lab-4 pipeline: sample+split the corpus, train the four
n-gram models with Laplace smoothing, evaluate perplexity, and draw charts.
"""
import build_splits
import train_eval
import visualize


def main():
    build_splits.main()
    train_eval.main()
    visualize.main()


if __name__ == '__main__':
    main()
