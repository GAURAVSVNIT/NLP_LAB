"""Plots dev perplexity vs. k (log-x) per n-gram order from output/k_sweep.json,
marking each order's dev-perplexity-minimizing k* against the assignment's
specified k=0.3."""
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

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
ORDERS = ['unigram', 'bigram', 'trigram', 'quadrigram']


def main():
    results = json.loads((OUTPUT_DIR / 'k_sweep.json').read_text(encoding='utf-8'))
    best = results['best_by_dev_perplexity']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, order, color in zip(axes.flat, ORDERS, PALETTE):
        points = sorted(results['sweep'][order], key=lambda p: p['k'])
        ks = [p['k'] for p in points]
        pps = [p['dev_perplexity'] for p in points]
        ax.plot(ks, pps, marker='o', markersize=3, color=color, linewidth=1.5)
        ax.set_xscale('log')
        ax.axvline(0.3, color='#999', linestyle='--', linewidth=1, label='Assignment k=0.3')
        k_star = best[order]['k']
        pp_star = best[order]['dev_perplexity']
        ax.scatter([k_star], [pp_star], color='#222', zorder=5, s=35, label=f'k*={k_star:.2g}')
        ax.set_title(order.capitalize())
        ax.set_xlabel('k (log scale)')
        ax.set_ylabel('Dev perplexity')
        ax.legend(fontsize=8)
    fig.suptitle('Dev perplexity vs. Add-k smoothing constant, by n-gram order')
    fig.tight_layout()
    fig.savefig(ASSETS / 'k_sweep.png', dpi=150)
    plt.close(fig)
    print('wrote assets/k_sweep.png')


if __name__ == '__main__':
    main()
