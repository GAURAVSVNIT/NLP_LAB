"""Comparison charts for the four POS decoders: per-tag F1 (dev and test) and
overall accuracy / macro-F1.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data import ASSETS, OUTPUT_DIR

METHOD_LABELS = {
    'naive_bayes': 'Naive Bayes',
    'hmm_forward_viterbi': 'HMM forward Viterbi',
    'hmm_backward_viterbi': 'HMM backward Viterbi',
    'hmm_forward_backward_viterbi': 'HMM forward-backward Viterbi',
}
PALETTE = {
    'naive_bayes': '#8C8C8C',
    'hmm_forward_viterbi': '#4C72B0',
    'hmm_backward_viterbi': '#DD8452',
    'hmm_forward_backward_viterbi': '#55A868',
}


def plot_per_tag_f1(results, split, out_path):
    methods = list(METHOD_LABELS)
    tags = sorted(results['corpus']['tag_counts_train'],
                  key=lambda t: results['corpus']['tag_counts_train'][t], reverse=True)
    n = len(tags)
    height = 0.2
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, 0.45 * n + 2))
    for k, m in enumerate(methods):
        f1 = [results['methods'][m][split]['per_tag'].get(t, {}).get('f1', 0.0) for t in tags]
        ax.barh(y + (k - 1.5) * height, f1, height, label=METHOD_LABELS[m], color=PALETTE[m])

    ax.set_yticks(y)
    support = results['methods']['naive_bayes'][split]['per_tag']
    ax.set_yticklabels([f"{t} (n={support[t]['support']})" for t in tags], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1-score')
    ax.set_title(f'Per-tag F1 on the Hindi {split} set (tags ordered by training frequency)')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overall(results, out_path):
    methods = list(METHOD_LABELS)
    x = np.arange(len(methods))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = [
        ('Accuracy (dev)', 'dev', 'accuracy', '#9DB4D6'),
        ('Accuracy (test)', 'test', 'accuracy', '#4C72B0'),
        ('Macro-F1 (dev)', 'dev', 'macro_f1', '#B6DDBF'),
        ('Macro-F1 (test)', 'test', 'macro_f1', '#55A868'),
    ]
    for k, (label, split, key, color) in enumerate(series):
        vals = [results['methods'][m][split][key] for m in methods]
        bars = ax.bar(x + (k - 1.5) * width, vals, width, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Overall POS tagging performance: all decoders')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    results = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    ASSETS.mkdir(exist_ok=True)
    plot_per_tag_f1(results, 'test', ASSETS / 'per_tag_f1_test.png')
    plot_per_tag_f1(results, 'dev', ASSETS / 'per_tag_f1_dev.png')
    plot_overall(results, ASSETS / 'overall_comparison.png')
    print(f'Charts written to {ASSETS}')


if __name__ == '__main__':
    main()
