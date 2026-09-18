"""Builds the charts referenced in the README/report from output/results.json,
contrasting Add-k (k=0.3) against Assignment 4's Laplace (k=1) baseline."""
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
ORDERS = ['unigram', 'bigram', 'trigram', 'quadrigram']


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=150)
    plt.close(fig)
    print(f'wrote assets/{name}')


def plot_addk_vs_laplace(results, split):
    pp = results['perplexity']
    addk = [pp[o][f'add_k_{split}_perplexity'] for o in ORDERS]
    laplace = [pp[o][f'laplace_{split}_perplexity'] for o in ORDERS]

    x = range(len(ORDERS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], laplace, width, label='Laplace (k=1, Lab 4)', color=PALETTE[3])
    ax.bar([i + width / 2 for i in x], addk, width, label=f'Add-k (k={results["k"]})', color=PALETTE[2])
    ax.set_xticks(list(x))
    ax.set_xticklabels([o.capitalize() for o in ORDERS])
    ax.set_ylabel(f'{split.capitalize()} perplexity (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'Add-k (k={results["k"]}) vs. Laplace smoothing -- {split} perplexity')
    for i, (l, a) in enumerate(zip(laplace, addk)):
        ax.text(i - width / 2, l, f'{l:,.0f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width / 2, a, f'{a:,.0f}', ha='center', va='bottom', fontsize=8)
    ax.legend()
    save(fig, f'addk_vs_laplace_{split}.png')


def plot_pct_improvement(results):
    pp = results['perplexity']
    dev_pct = [100 * (pp[o]['laplace_dev_perplexity'] - pp[o]['add_k_dev_perplexity']) / pp[o]['laplace_dev_perplexity'] for o in ORDERS]
    test_pct = [100 * (pp[o]['laplace_test_perplexity'] - pp[o]['add_k_test_perplexity']) / pp[o]['laplace_test_perplexity'] for o in ORDERS]

    x = range(len(ORDERS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], dev_pct, width, label='Dev', color=PALETTE[0])
    ax.bar([i + width / 2 for i in x], test_pct, width, label='Test', color=PALETTE[1])
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([o.capitalize() for o in ORDERS])
    ax.set_ylabel('Perplexity reduction vs. Laplace (%)')
    ax.set_title(f'Add-k (k={results["k"]}) improvement over Laplace, by order')
    for i, (d, t) in enumerate(zip(dev_pct, test_pct)):
        ax.text(i - width / 2, d, f'{d:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.text(i + width / 2, t, f'{t:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.legend()
    save(fig, 'pct_improvement.png')


def main():
    results = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    plot_addk_vs_laplace(results, 'dev')
    plot_addk_vs_laplace(results, 'test')
    plot_pct_improvement(results)


if __name__ == '__main__':
    main()
