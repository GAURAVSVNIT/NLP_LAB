"""
Generates the visual-analysis charts for the README from the stats/
analysis JSON files produced by compute_stats.py, process_full_corpus.py,
export_parquet_partial.py and export_parquet_full.py.

Run this after the corresponding data-generation scripts have produced
their JSON outputs. Missing full-corpus files are skipped gracefully so
this can be run once with just the partial results and re-run later once
the full-corpus job finishes.
"""
import json
import os

import matplotlib
matplotlib.rcParams["font.family"] = "Nirmala UI"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

BASE = r"D:\CODING\NLP\Assignment_1"
ASSETS = os.path.join(BASE, "assets")
os.makedirs(ASSETS, exist_ok=True)

PARTIAL_STATS = os.path.join(BASE, "output", "corpus_statistics.json")
FULL_STATS = os.path.join(BASE, "output_full", "corpus_statistics.json")
PARTIAL_ANALYSIS = os.path.join(BASE, "output", "analysis_partial.json")
FULL_ANALYSIS = os.path.join(BASE, "output_full", "analysis_full.json")

COLOR_PARTIAL = "#4C72B0"
COLOR_FULL = "#DD8452"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def chart_corpus_size_comparison(partial, full):
    metrics = ["total_sentences", "total_words", "total_characters"]
    labels = ["Sentences", "Words", "Characters"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(metrics))
    width = 0.35

    partial_vals = [partial[m] for m in metrics]
    ax.bar([i - width / 2 for i in x], partial_vals, width, label=f"Partial sample ({partial['total_sentences']:,} sent.)", color=COLOR_PARTIAL)

    if full is not None:
        full_vals = [full[m] for m in metrics]
        ax.bar([i + width / 2 for i in x], full_vals, width, label=f"Full corpus ({full['total_sentences']:,} sent.)", color=COLOR_FULL)

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Corpus Size: Partial Sample vs. Full Dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "corpus_size_comparison.png")


def chart_avg_length_ttr(partial, full):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    specs = [
        ("average_sentence_length", "Avg. Sentence Length\n(words/sentence)"),
        ("average_word_length", "Avg. Word Length\n(chars/word)"),
        ("type_token_ratio", "Type/Token Ratio\n(unique/total)"),
    ]
    for ax, (key, title) in zip(axes, specs):
        vals = [partial[key]]
        labels = ["Partial"]
        colors = [COLOR_PARTIAL]
        if full is not None:
            vals.append(full[key])
            labels.append("Full")
            colors.append(COLOR_FULL)
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.margins(y=0.15)
    fig.suptitle("Average Lengths and Type/Token Ratio")
    fig.tight_layout()
    save(fig, "avg_length_ttr_comparison.png")


def chart_token_type_distribution(partial_analysis):
    dist = partial_analysis["token_type_distribution"]
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=PALETTE[: len(labels)])
    ax.set_yscale("log")
    ax.set_ylabel("Token count (log scale)")
    ax.set_title("Token Type Distribution (partial sample, 7.8M tokens)")
    pct_labels = [f"{v/total*100:.2f}%" for v in values]
    ax.bar_label(bars, labels=pct_labels, padding=3, fontsize=8)
    fig.tight_layout()
    save(fig, "token_type_distribution.png")


def chart_sentence_length_distribution(partial_analysis, full_analysis):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def to_xy(dist, cap=None):
        xs = sorted(int(k) for k in dist)
        ys = [dist[str(x)] for x in xs]
        total = sum(ys)
        ys_pct = [y / total * 100 for y in ys]
        return xs, ys_pct

    xs_p, ys_p = to_xy(partial_analysis["sentence_length_distribution"])
    max_x = 40
    xs_p_c = [x for x in xs_p if x <= max_x]
    ys_p_c = [y for x, y in zip(xs_p, ys_p) if x <= max_x]
    ax.plot(xs_p_c, ys_p_c, marker="o", markersize=3, label="Partial sample", color=COLOR_PARTIAL)

    if full_analysis is not None:
        xs_f, ys_f = to_xy(full_analysis["sentence_length_distribution"])
        xs_f_c = [x for x in xs_f if x <= max_x]
        ys_f_c = [y for x, y in zip(xs_f, ys_f) if x <= max_x]
        ax.plot(xs_f_c, ys_f_c, marker="o", markersize=3, label="Full corpus", color=COLOR_FULL)

    ax.set_xlabel("Sentence length (words)")
    ax.set_ylabel("% of sentences")
    ax.set_title("Sentence Length Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "sentence_length_distribution.png")


def chart_top_words(analysis, key_suffix, title):
    top = analysis["top_50_words"][:20]
    words = [w for w, _ in top][::-1]
    counts = [c for _, c in top][::-1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(words, counts, color=COLOR_PARTIAL if key_suffix == "partial" else COLOR_FULL)
    ax.set_xlabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()
    save(fig, f"top20_words_{key_suffix}.png")


def main():
    partial_stats = load(PARTIAL_STATS)
    full_stats = load(FULL_STATS)
    partial_analysis = load(PARTIAL_ANALYSIS)
    full_analysis = load(FULL_ANALYSIS)

    if partial_stats:
        chart_corpus_size_comparison(partial_stats, full_stats)
        chart_avg_length_ttr(partial_stats, full_stats)

    if partial_analysis:
        chart_token_type_distribution(partial_analysis)
        chart_sentence_length_distribution(partial_analysis, full_analysis)
        chart_top_words(partial_analysis, "partial", "Top 20 Most Frequent Words (partial sample)")

    if full_analysis:
        chart_top_words(full_analysis, "full", "Top 20 Most Frequent Words (full corpus)")


if __name__ == "__main__":
    main()
