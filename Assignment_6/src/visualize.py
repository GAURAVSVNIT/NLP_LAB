"""Generates publication-quality comparison charts for Assignment 6
evaluating the 5 smoothing techniques against Assignment 4 (Laplace)
and Assignment 5 (Add-k) baselines.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.rcParams['font.family'] = 'Nirmala UI'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)

ORDERS = ['unigram', 'bigram', 'trigram', 'quadrigram']
METHOD_LABELS = {
    'laplace': 'Laplace (Lab 4)',
    'add_k': 'Add-k (k=0.3, Lab 5)',
    'good_turing': 'Good-Turing',
    'stupid_backoff': 'Stupid Backoff (alpha=0.4)',
    'katz_backoff': 'Katz Backoff',
    'interpolated': 'Interpolated (Jelinek-Mercer)',
    'kneser_ney': 'Kneser-Ney (Interpolated)',
}

PALETTE = {
    'laplace': '#C44E52',        # Red
    'add_k': '#DD8452',          # Orange
    'good_turing': '#8C8C8C',    # Grey
    'stupid_backoff': '#DA8BC3', # Magenta
    'katz_backoff': '#8172B2',   # Purple
    'interpolated': '#4C72B0',   # Blue
    'kneser_ney': '#55A868',     # Green
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=160)
    plt.close(fig)
    print(f'Wrote assets/{name}')


def plot_grouped_bar(results, split='test'):
    """Grouped bar chart comparing all smoothing techniques across orders on log-scale."""
    by_order = results['by_order']
    methods = ['laplace', 'add_k', 'good_turing', 'stupid_backoff', 'katz_backoff', 'interpolated', 'kneser_ney']
    
    x = np.arange(len(ORDERS))
    width = 0.11
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    for i, m in enumerate(methods):
        vals = []
        for o in ORDERS:
            val = by_order[o].get(m, {}).get(split, None)
            vals.append(val if val is not None else 1.0)
        
        offset = (i - len(methods) / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=METHOD_LABELS[m], color=PALETTE[m])

    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in ORDERS], fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{split.capitalize()} Perplexity (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title(f'Comparison of Smoothing Techniques on {split.capitalize()} Set by N-gram Order', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.3, axis='y')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), framealpha=0.9, fontsize=9)
    
    save(fig, f'smoothing_comparison_{split}.png')


def plot_scaling_trend(results):
    """Line plot showing perplexity scaling with order (1 to 4)."""
    by_order = results['by_order']
    methods = ['laplace', 'add_k', 'good_turing', 'katz_backoff', 'interpolated', 'kneser_ney']
    
    fig, ax = plt.subplots(figsize=(9, 5.2))
    
    orders_num = [1, 2, 3, 4]
    
    for m in methods:
        vals = []
        for o in ORDERS:
            val = by_order[o].get(m, {}).get('test', None)
            vals.append(val)
        
        # Style lines: dashed for baselines, solid for advanced
        linestyle = '--' if m in ('laplace', 'add_k', 'good_turing') else '-'
        marker = 's' if m in ('laplace', 'add_k') else 'o'
        ax.plot(orders_num, vals, marker=marker, linewidth=2.2, linestyle=linestyle,
                label=METHOD_LABELS[m], color=PALETTE[m])
        
        # Add annotation on test value for quadgram
        if vals[-1] is not None:
            ax.annotate(f'{vals[-1]:,.0f}', (4, vals[-1]),
                        textcoords="offset points", xytext=(8, -2), ha='left', fontsize=8,
                        color=PALETTE[m], fontweight='bold')

    ax.set_xticks(orders_num)
    ax.set_xticklabels(['Unigram (n=1)', 'Bigram (n=2)', 'Trigram (n=3)', 'Quadgram (n=4)'], fontsize=10)
    ax.set_ylabel('Test Perplexity (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Perplexity Trajectory vs. N-gram Order: Additive Over-Smoothing vs. Backoff / Interpolation', fontsize=11, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(0.02, 0.45), framealpha=0.9, fontsize=9.5)
    
    save(fig, 'order_scaling_trend.png')


def plot_kneser_ney_vs_baselines(results):
    """Direct comparison between Laplace, Add-k, and Kneser-Ney."""
    by_order = results['by_order']
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    x = np.arange(len(ORDERS))
    width = 0.25
    
    laplace = [by_order[o]['laplace']['test'] for o in ORDERS]
    add_k = [by_order[o]['add_k']['test'] for o in ORDERS]
    kn = [by_order[o]['kneser_ney']['test'] for o in ORDERS]
    
    ax.bar(x - width, laplace, width, label='Laplace (k=1, Lab 4)', color=PALETTE['laplace'])
    ax.bar(x, add_k, width, label='Add-k (k=0.3, Lab 5)', color=PALETTE['add_k'])
    ax.bar(x + width, kn, width, label='Kneser-Ney (Lab 6)', color=PALETTE['kneser_ney'])
    
    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in ORDERS], fontsize=10, fontweight='bold')
    ax.set_ylabel('Test Perplexity (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Kneser-Ney Smoothing vs. Additive Baselines (Test Perplexity)', fontsize=11, fontweight='bold')
    
    for i in range(len(ORDERS)):
        ax.text(i - width, laplace[i], f'{laplace[i]:,.0f}', ha='center', va='bottom', fontsize=7.5)
        ax.text(i, add_k[i], f'{add_k[i]:,.0f}', ha='center', va='bottom', fontsize=7.5)
        ax.text(i + width, kn[i], f'{kn[i]:,.0f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        
    ax.grid(True, which='both', linestyle='--', alpha=0.3, axis='y')
    ax.legend(framealpha=0.9)
    
    save(fig, 'kneser_ney_vs_baselines.png')


def main():
    results = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    plot_grouped_bar(results, 'dev')
    plot_grouped_bar(results, 'test')
    plot_scaling_trend(results)
    plot_kneser_ney_vs_baselines(results)


if __name__ == '__main__':
    main()
