"""Charts for Lab 8: the FST state diagram, the TF-IDF similarity matrix and the spell-corrector
comparisons. Everything is written to assets/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)

INK, MUTED, GRID, ORACLE = '#0b0b0b', '#52514e', '#e4e3df', '#b9b8b2'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
BLUES = LinearSegmentedColormap.from_list('blues', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
MODE_LABELS = {'prior': 'Prior only  P(c)', 'channel': 'Channel only  P(x|c)',
               'noisy_joint': 'Noisy channel (joint P(edit))',
               'noisy_conditional': 'Noisy channel (conditional P(edit))'}
MODES = list(MODE_LABELS)

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


# ----------------------------------------------------------------- midsem figures
def plot_fst():
    pos = {'q0': (0, 0), 'q1': (3.2, 3.0), 'q4': (3.2, 1.0), 'q2': (3.2, -1.0), 'q3': (3.2, -3.0),
           'q5': (6.0, 0.0), 'q6': (9.0, 2.2), 'q7': (9.0, -2.2), 'q8': (12.0, 0)}
    fill = {'q0': '#cde2fb', 'q1': '#ffffff', 'q4': '#ffffff', 'q2': '#fde3d6', 'q3': '#fde3d6',
            'q5': '#fde3d6', 'q6': '#ffffff', 'q7': '#ffffff', 'q8': '#d1e7dd'}
    fig, ax = plt.subplots(figsize=(13, 7))
    r = 0.45

    def edge(a, b, label, rad=0.0, lx=0.0, ly=0.0, color=INK):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
                                     connectionstyle=f'arc3,rad={rad}', shrinkA=r * 72 / 1.6,
                                     shrinkB=r * 72 / 1.6, color=color, linewidth=1.3))
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha='center', va='center', fontsize=9,
                color=INK, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))

    edge('q0', 'q1', 'other, h : copy')
    edge('q0', 'q4', 'c : c')
    edge('q0', 'q2', 's : s')
    edge('q0', 'q3', 'x, z : copy')
    edge('q4', 'q5', 'h : h')
    edge('q2', 'q5', 'h : h')
    edge('q1', 'q6', '+N : ε')
    edge('q4', 'q6', '+N : ε')
    edge('q2', 'q7', '+N : ε')
    edge('q3', 'q7', '+N : ε')
    edge('q5', 'q7', '+N : ε')
    edge('q6', 'q8', '+SG : ε\n+PL : s')
    edge('q7', 'q8', '+SG : ε\n+PL : es', color='#c0392b')

    for q, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), r, fc=fill[q], ec=INK, lw=1.4))
        if q == 'q8':
            ax.add_patch(Circle((x, y), r * 0.82, fc='none', ec=INK, lw=1.2))
        ax.text(x, y, q, ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(pos['q0'][0] - r, 0), xytext=(-1.4, 0),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3))
    ax.text(-1.45, 0.25, 'start', fontsize=9, color=MUTED)

    note = ('Letter arcs from every stem state q1-q5 (each copies the letter, l:l):\n'
            '  s -> q2    x, z -> q3    c -> q4    h -> q5 if in q2 or q4, else q1    any other letter -> q1\n'
            'Orange states = stem ends in a sibilant (s, x, z, ch, sh) -> plural "es".   q8 = accepting state.')
    ax.text(5.6, -4.55, note, ha='center', va='center', fontsize=9, color=INK,
            bbox=dict(boxstyle='round,pad=0.5', fc='#f4f5f7', ec=GRID))
    ax.set_xlim(-1.8, 12.8)
    ax.set_ylim(-5.4, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('FST for English noun inflection (lexical: stem +N +SG/+PL  ->  surface form)')
    save(fig, 'q1_fst_diagram.png')


def plot_tfidf(sol):
    q3 = sol['q3_tfidf']
    n = len(q3['sentences'])
    m = np.eye(n)
    for p in q3['pairs']:
        i, j = (int(x[1:]) - 1 for x in p['pair'].split('-'))
        m[i, j] = m[j, i] = p['cosine']
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(m, cmap=BLUES, vmin=0, vmax=1)
    labels = [f'S{i + 1}: {s}' for i, s in enumerate(q3['sentences'])]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{m[i, j]:.3f}', ha='center', va='center', fontsize=10,
                    color='white' if m[i, j] > 0.6 else INK)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f'S{i + 1}' for i in range(n)])
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label='Cosine similarity')
    ax.set_title('Q3: TF-IDF cosine similarity', fontsize=11)
    save(fig, 'q3_tfidf_cosine.png')


# ----------------------------------------------------------------- spell-corrector figures
def plot_spell_accuracy(res):
    fig, ax = plt.subplots(figsize=(11, 5))
    groups = ['d1', 'd2']
    x = np.arange(len(groups))
    width = 0.16
    for i, mode in enumerate(MODES):
        vals = [res['main'][f'detected|{g}|{mode}']['accuracy'] * 100 for g in groups]
        bars = ax.bar(x + (i - 2) * width, vals, width * 0.92, color=SERIES[i], label=MODE_LABELS[mode])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f'{v:.1f}', ha='center', fontsize=8, color=MUTED)
    vals = [res['oracle'][g] * 100 for g in groups]
    bars = ax.bar(x + 2 * width, vals, width * 0.92, color=ORACLE, hatch='//', edgecolor='white',
                  label='Oracle: correct word among candidates')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f'{v:.1f}', ha='center', fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(['Candidates within 1 edit', 'Candidates within 2 edits'], fontsize=11)
    ax.set_ylabel('Top-1 accuracy over all 39,507 pairs (%)')
    ax.set_ylim(0, 70)
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    ax.set_title('Spell correction accuracy by scoring model (error detection given)')
    ax.legend(fontsize=8.5, loc='upper left', ncol=2)
    save(fig, 'spell_accuracy.png')


def plot_breakdown(res):
    configs = [('detected|d1|noisy_joint', 'd=1, joint'), ('detected|d1|noisy_conditional', 'd=1, conditional'),
               ('detected|d2|noisy_joint', 'd=2, joint'), ('detected|d2|noisy_conditional', 'd=2, conditional')]
    group_order = ['distance=1 non-word (classic setting)', 'correct word reachable', 'distance=1',
                   'distance=2', 'distance=3+', 'typo is a non-word', 'typo is a vocabulary word']
    group_labels = ['Distance-1\nnon-word', 'Correct word\nreachable', 'True distance\n= 1',
                    'True distance\n= 2', 'True distance\n>= 3', 'Typo is a\nnon-word', 'Typo is a\nvocab word']
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(group_order))
    width = 0.2
    for i, (key, label) in enumerate(configs):
        vals = [res['breakdown'][key][g]['accuracy'] * 100 for g in group_order]
        ax.bar(x + (i - 1.5) * width, vals, width * 0.92, color=SERIES[i], label=f'Noisy channel, {label}')
    ns = [res['breakdown'][configs[0][0]][g]['n_pairs'] for g in group_order]
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lbl}\n(n={n:,})' if i != 1 else lbl for i, (lbl, n) in enumerate(zip(group_labels, ns))],
                       fontsize=8.5)
    ax.set_ylabel('Top-1 accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    ax.set_title('Accuracy by error type (error detection given)')
    ax.legend(fontsize=8.5, ncol=4, loc='upper right')
    save(fig, 'spell_breakdown.png')


def plot_p_err(res):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    styles = {'d1|joint': (SERIES[0], '-'), 'd2|joint': (SERIES[0], '--'),
              'd1|conditional': (SERIES[1], '-'), 'd2|conditional': (SERIES[1], '--')}
    for key, row in res['p_err_sweep'].items():
        xs = [float(k) for k in row]
        color, ls = styles[key]
        d, ch = key.split('|')
        ax.plot(xs, [v * 100 for v in row.values()], marker='o', color=color, linestyle=ls, linewidth=2,
                markersize=6, label=f'{ch} P(edit), candidates within {d[1]} edit{"s" if d == "d2" else ""}')
    for d in ['d1', 'd2']:
        v = res['main'][f'detected|{d}|noisy_joint']['accuracy'] * 100
        ax.axhline(v, color=MUTED, linestyle=':', linewidth=1)
        ax.text(1e-4, v + 0.35, f'reference: detection given, {d[1]} edit{"s" if d == "d2" else ""}, joint = {v:.1f}%',
                fontsize=8, color=MUTED)
    ax.set_xscale('log')
    ax.set_xlabel('p_err  (prior probability that the input is misspelled)')
    ax.set_ylabel('Top-1 accuracy (%)')
    ax.grid(color=GRID)
    ax.set_title('Open setting: the typo itself is a candidate', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=3.5)
    save(fig, 'spell_p_err_sweep.png')


def plot_cv(res):
    cv = res['cross_validation']
    configs = ['d1|joint', 'd1|conditional', 'd2|joint', 'd2|conditional']
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(configs))
    width = 0.36
    for i, (src, label) in enumerate([('count_1edit', 'count_1edit.txt (built from all errors)'),
                                      ('held_out', 'Edit counts re-estimated on training folds only')]):
        vals = [cv[f'{src}|{c}']['mean'] * 100 for c in configs]
        bars = ax.bar(x + (i - 0.5) * width, vals, width * 0.92, color=SERIES[i * 2], label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f'{v:.1f}', ha='center', fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={c[1]}, {c.split("|")[1]}' for c in configs])
    ax.set_ylabel('5-fold CV top-1 accuracy (%)')
    ax.set_ylim(0, 50)
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    ax.set_title('Leakage check: channel counts from held-out folds', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper left')
    save(fig, 'spell_cv_leakage.png')


def main():
    sol = json.loads((OUTPUT_DIR / 'midsem_solutions.json').read_text(encoding='utf-8'))
    plot_fst()
    plot_tfidf(sol)
    spell_path = OUTPUT_DIR / 'spell_results.json'
    if spell_path.exists():
        res = json.loads(spell_path.read_text(encoding='utf-8'))
        plot_spell_accuracy(res)
        plot_breakdown(res)
        plot_p_err(res)
        plot_cv(res)


if __name__ == '__main__':
    main()
