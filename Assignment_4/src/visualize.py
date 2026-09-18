"""Builds the charts referenced in the README from output/results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.rcParams['font.family'] = 'Nirmala UI'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974']


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=150)
    plt.close(fig)
    print(f'wrote assets/{name}')


def plot_perplexity(results):
    pp = results['perplexity']
    orders = ['unigram', 'bigram', 'trigram', 'quadrigram']
    dev = [pp[o]['dev_perplexity'] for o in orders]
    test = [pp[o]['test_perplexity'] for o in orders]

    x = range(len(orders))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], dev, width, label='Dev', color=PALETTE[0])
    ax.bar([i + width / 2 for i in x], test, width, label='Test', color=PALETTE[1])
    ax.set_xticks(list(x))
    ax.set_xticklabels([o.capitalize() for o in orders])
    ax.set_ylabel('Perplexity (Laplace-smoothed, log scale)')
    ax.set_yscale('log')
    ax.set_title('Held-out perplexity by n-gram order')
    for i, (d, t) in enumerate(zip(dev, test)):
        ax.text(i - width / 2, d, f'{d:.0f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width / 2, t, f'{t:.0f}', ha='center', va='bottom', fontsize=8)
    ax.legend()
    save(fig, 'perplexity_by_order.png')


def plot_vocab_oov(results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    labels = ['Raw types\n(train)', 'Kept vocab\n(+<UNK>, +</s>)']
    values = [results['raw_train_types'], results['vocab_size']]
    ax.bar(labels, values, color=[PALETTE[2], PALETTE[0]])
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    ax.set_title('Vocabulary size (min_count=2 collapses\nsingletons to <UNK>)')
    ax.set_ylabel('Word types')

    ax = axes[1]
    labels = ['Dev', 'Test']
    values = [results['dev_oov_rate'] * 100, results['test_oov_rate'] * 100]
    ax.bar(labels, values, color=[PALETTE[3], PALETTE[4]])
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.set_title('Out-of-vocabulary rate\n(tokens mapped to <UNK>)')
    ax.set_ylabel('% of tokens')

    save(fig, 'vocab_and_oov.png')


def plot_ngram_growth(results):
    pp = results['perplexity']
    orders = ['unigram', 'bigram', 'trigram', 'quadrigram']
    counts = [pp[o]['n_unique_ngrams'] for o in orders]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(orders, counts, color=PALETTE[5])
    ax.set_yscale('log')
    ax.set_ylabel('Unique n-grams seen in training (log scale)')
    ax.set_title('n-gram sparsity: distinct n-grams explode with order')
    for i, v in enumerate(counts):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
    save(fig, 'ngram_growth.png')


def main():
    results = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    plot_perplexity(results)
    plot_vocab_oov(results)
    plot_ngram_growth(results)


if __name__ == '__main__':
    main()
