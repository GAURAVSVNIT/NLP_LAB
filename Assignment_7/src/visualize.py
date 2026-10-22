"""Generates the comparison charts for Lab 7 into assets/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)

ALGORITHMS = ['naive_bayes', 'forward_viterbi', 'backward_viterbi', 'forward_backward_viterbi']
LABELS = {
    'naive_bayes': 'Naive Bayes',
    'forward_viterbi': 'HMM forward Viterbi',
    'backward_viterbi': 'HMM backward Viterbi',
    'forward_backward_viterbi': 'HMM forward-backward Viterbi',
}
PALETTE = {
    'naive_bayes': '#2a78d6',
    'forward_viterbi': '#eb6834',
    'backward_viterbi': '#1baf7a',
    'forward_backward_viterbi': '#eda100',
}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUES = LinearSegmentedColormap.from_list('blues', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.titleweight': 'bold', 'axes.titlesize': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=160, facecolor='white')
    plt.close(fig)
    print(f'Wrote assets/{name}')


def tags_by_frequency(results):
    return list(results['train_tag_counts'])


def plot_per_tag_f1(results, split):
    tags = tags_by_frequency(results)
    res = results['splits'][split]
    x = np.arange(len(tags))
    width = 0.2
    fig, ax = plt.subplots(figsize=(15, 5.6))
    for i, a in enumerate(ALGORITHMS):
        vals = [res[a]['per_tag'][t]['f1'] for t in tags]
        ax.bar(x + (i - 1.5) * width, vals, width * 0.92, label=LABELS[a], color=PALETTE[a])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={res[ALGORITHMS[0]]['per_tag'][t]['support']})" for t in tags],
                       rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('F1-score')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f'Per-tag F1 on the {split} split (tags ordered by training frequency)')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.32), fontsize=9)
    save(fig, f'per_tag_f1_{split}.png')


def plot_f1_heatmap(results, split):
    tags = tags_by_frequency(results)
    res = results['splits'][split]
    m = np.array([[res[a]['per_tag'][t]['f1'] for a in ALGORITHMS] for t in tags])
    fig, ax = plt.subplots(figsize=(7.6, 9.5))
    ax.imshow(m, cmap=BLUES, vmin=0.5, vmax=1, aspect='auto')  # values below 0.5 clip to the lightest step
    for i in range(len(tags)):
        for j in range(len(ALGORITHMS)):
            v = m[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if v > 0.83 else INK)
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels([LABELS[a].replace('HMM ', 'HMM\n') for a in ALGORITHMS], fontsize=8.5)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([f"{t} ({res[ALGORITHMS[0]]['per_tag'][t]['support']})" for t in tags], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'Per-tag F1 heat map, {split} split (colour scale 0.5 to 1.0)')
    save(fig, f'f1_heatmap_{split}.png')


def plot_overall(results):
    metrics = [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1'), ('weighted_f1', 'Weighted F1'),
               ('oov_accuracy', 'OOV-word accuracy')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, split in zip(axes, ['dev', 'test']):
        res = results['splits'][split]
        x = np.arange(len(metrics))
        width = 0.2
        for i, a in enumerate(ALGORITHMS):
            vals = [res[a][m] for m, _ in metrics]
            bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.92, label=LABELS[a], color=PALETTE[a])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v * 100:.1f}', ha='center',
                        va='bottom', fontsize=6.5, color=MUTED, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in metrics])
        ax.set_ylim(0, 1.12)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f'{split.capitalize()} split')
    axes[0].set_ylabel('Score')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ASSETS / 'overall_comparison.png', dpi=160, facecolor='white')
    plt.close(fig)
    print('Wrote assets/overall_comparison.png')


def plot_confusion(split):
    conf = json.loads((OUTPUT_DIR / f'confusion_{split}.json').read_text(encoding='utf-8'))
    tags = conf['tags']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6))
    for ax, key, title in [(axes[0], 'naive_bayes_matrix', 'Naive Bayes'),
                           (axes[1], 'matrix', LABELS[conf['algorithm']])]:
        m = np.array(conf[key], dtype=float)
        rows = m.sum(axis=1, keepdims=True)
        norm = np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)
        im = ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=90, fontsize=7)
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(tags, fontsize=7)
        ax.set_xlabel('Predicted tag')
        ax.set_ylabel('Gold tag')
        ax.set_title(f'{title}: row-normalised confusion ({split})', fontsize=11)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label='Share of gold-tag tokens')
    fig.savefig(ASSETS / f'confusion_{split}.png', dpi=160, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote assets/confusion_{split}.png')


def plot_tuning(results):
    tuning = results['tuning_dev']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    alphas = list(tuning['nb_alpha'])
    ax = axes[0]
    ax.plot(range(len(alphas)), [tuning['nb_alpha'][a] * 100 for a in alphas], marker='o',
            color=PALETTE['naive_bayes'], linewidth=2, markersize=8)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels(alphas)
    ax.set_xlabel('Laplace alpha')
    ax.set_ylabel('Dev accuracy (%)')
    ax.set_title('Naive Bayes: smoothing sweep (dev)')
    ax.grid(color=GRID, linewidth=0.8)

    ks = list(tuning['hmm_k'])
    ax = axes[1]
    for a in ALGORITHMS[1:]:
        style = {'linestyle': '--', 'marker': 's'} if a == 'backward_viterbi' else {'marker': 'o'}
        ax.plot(range(len(ks)), [tuning['hmm_k'][k][a] * 100 for k in ks], color=PALETTE[a],
                linewidth=2, markersize=8, label=LABELS[a], **style)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_xlabel('Transition add-k')
    ax.set_title('HMM: transition smoothing sweep (dev)')
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(fontsize=8.5, loc='lower left')
    save(fig, 'tuning_dev.png')


def main():
    results = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    for split in ['dev', 'test']:
        plot_per_tag_f1(results, split)
        plot_f1_heatmap(results, split)
    plot_overall(results)
    plot_confusion('test')
    plot_tuning(results)


if __name__ == '__main__':
    main()
