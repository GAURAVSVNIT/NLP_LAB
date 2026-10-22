"""Builds U24AI026_NLP_LAB-7.pdf: method, formulations, per-tag F1 tables, charts and analysis for the
Hindi POS tagging experiments. All numbers are read from output/results.json.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'
REPORT_PATH = ROOT / 'U24AI026_NLP_LAB-7.pdf'

NAVY = colors.HexColor('#2C3E50')
BLUE = colors.HexColor('#2a78d6')
GREEN = colors.HexColor('#1baf7a')
ORANGE = colors.HexColor('#eb6834')
PURPLE = colors.HexColor('#6250d6')
BEST_BG = colors.HexColor('#D1E7DD')
GREY_BG = colors.HexColor('#F4F5F7')

ALGORITHMS = ['naive_bayes', 'forward_viterbi', 'backward_viterbi', 'forward_backward_viterbi']
SHORT = {'naive_bayes': 'Naive Bayes', 'forward_viterbi': 'Fwd Viterbi',
         'backward_viterbi': 'Bwd Viterbi', 'forward_backward_viterbi': 'Fwd-Bwd Viterbi'}
TAG_NAMES = {
    'N_NN': 'common noun', 'N_NNP': 'proper noun', 'N_NST': 'spatio-temporal noun', 'PSP': 'postposition',
    'V_VM': 'main verb', 'V_VAUX': 'auxiliary verb', 'JJ': 'adjective', 'RB': 'adverb',
    'PR_PRP': 'personal pronoun', 'PR_PRF': 'reflexive pronoun', 'PR_PRL': 'relative pronoun',
    'PR_PRQ': 'wh-pronoun', 'DM_DMD': 'deictic demonstrative', 'QT_QTC': 'cardinal number',
    'QT_QTF': 'general quantifier', 'QT_QTO': 'ordinal number', 'CC_CCD': 'coordinating conj.',
    'CC_CCS': 'subordinating conj.', 'RP_RPD': 'default particle', 'RP_NEG': 'negation particle',
    'RP_INTF': 'intensifier', 'RP_INJ': 'interjection', 'RD_PUNC': 'punctuation', 'RD_SYM': 'symbol',
}

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('ReportTitle', parent=styles['Title'], fontSize=21, textColor=NAVY, spaceAfter=4))
styles.add(ParagraphStyle('ReportSubtitle', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#555'), alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, textColor=NAVY, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle('H3', parent=styles['Heading3'], fontSize=10.5, textColor=NAVY, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9.5, leading=13.5, spaceAfter=6))
styles.add(ParagraphStyle('Formula', parent=styles['Code'], fontSize=8.5, leading=11.5, alignment=TA_LEFT, textColor=NAVY, backColor=GREY_BG, borderPadding=6, spaceBefore=2, spaceAfter=8))
styles.add(ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#666'), alignment=TA_CENTER, spaceBefore=3, spaceAfter=10))
styles.add(ParagraphStyle('Cell', parent=styles['Normal'], fontSize=7.5, leading=9))


def stat_card(value, label, color):
    t = Table([[Paragraph(f'<font color="{color.hexval()}" size="16"><b>{value}</b></font>')],
               [Paragraph(f'<font size="8" color="#555">{label}</font>')]],
              colWidths=[4.2 * cm])
    t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#DDD')),
        ('BACKGROUND', (0, 0), (-1, -1), GREY_BG),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 8),
    ]))
    return t


def base_table_style(n_rows):
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    for i in range(1, n_rows):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), GREY_BG))
    return style


def figure(name, caption, width=17.0, ratio=None):
    from reportlab.lib.utils import ImageReader
    path = ASSETS / name
    iw, ih = ImageReader(str(path)).getSize()
    ratio = ratio or ih / iw
    return KeepTogether([Image(str(path), width=width * cm, height=width * cm * ratio),
                         Paragraph(caption, styles['Caption'])])


def pct(x):
    return f'{x * 100:.2f}'


def build():
    r = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    splits = r['splits']
    test, dev = splits['test'], splits['dev']
    hp = r['hyperparameters']
    tags = list(r['train_tag_counts'])

    doc = SimpleDocTemplate(str(REPORT_PATH), pagesize=A4, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm)
    story = []

    # --- Title ---
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph('NLP Laboratory — Assignment 7', styles['ReportTitle']))
    story.append(Paragraph('Hindi Part-of-Speech Tagging: Naive Bayes and HMM (Forward, Backward and '
                           'Forward-Backward Viterbi)', styles['ReportSubtitle']))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#DDD')))
    story.append(Spacer(1, 0.4 * cm))
    meta = Table([
        ['Name', 'GAURAV DAMOR', 'Course', 'Natural Language Processing'],
        ['Roll No.', 'U24AI026', 'Lab', 'Lab 7 — Hindi POS Tagging'],
    ], colWidths=[2.2 * cm, 5.8 * cm, 2.0 * cm, 6.8 * cm])
    meta.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), NAVY), ('TEXTCOLOR', (2, 0), (2, -1), NAVY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta)
    story.append(Spacer(1, 0.5 * cm))

    best = max(ALGORITHMS, key=lambda a: test[a]['accuracy'])
    cards = Table([[
        stat_card(f'{r["corpus"]["train"]["tokens"]:,}', 'Training tokens', BLUE),
        stat_card(str(len(tags)), 'POS tags (BIS tagset)', PURPLE),
        stat_card(f'{pct(test[best]["accuracy"])}%', f'Best test accuracy ({SHORT[best]})', GREEN),
        stat_card(f'{pct(r["corpus"]["test"]["oov_rate"])}%', 'Test OOV rate', ORANGE),
    ]], colWidths=[4.25 * cm] * 4)
    cards.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story.append(cards)
    story.append(Spacer(1, 0.4 * cm))

    # --- 1. Task & data ---
    story.append(Paragraph('1. Task and Data', styles['H2']))
    story.append(Paragraph(
        'The goal is to assign a part-of-speech tag to every token of a Hindi sentence. The corpus is supplied '
        'as three tab-separated files (<i>word, tag</i> per line, blank line between sentences) using the BIS '
        'tagset. All probabilities are estimated <b>only from the training split</b>. The development split is '
        'used to choose the two smoothing constants, and the test split is used once, for final reporting. '
        'Four tagging algorithms are compared: a token-level <b>Naive Bayes</b> classifier and a bigram '
        '<b>Hidden Markov Model</b> decoded with <b>forward Viterbi</b>, <b>backward Viterbi</b> and '
        '<b>forward-backward Viterbi</b>.', styles['Body']))
    rows = [['Split', 'Sentences', 'Tokens', 'OOV rate (w.r.t. train vocabulary)']]
    for s in ['train', 'dev', 'test']:
        c = r['corpus'][s]
        rows.append([s.capitalize(), f'{c["sentences"]:,}', f'{c["tokens"]:,}',
                     f'{pct(c["oov_rate"])}%' if 'oov_rate' in c else f'— (vocabulary = {r["vocab_size"]:,} types)'])
    t = Table(rows, colWidths=[2.5 * cm, 3 * cm, 3 * cm, 7 * cm])
    t.setStyle(TableStyle(base_table_style(len(rows))))
    story.append(t)
    story.append(Paragraph('Table 1: Corpus statistics.', styles['Caption']))

    # --- 2. Methods ---
    story.append(Paragraph('2. Methods and Mathematical Formulation', styles['H2']))
    story.append(Paragraph('2.1 Naive Bayes classifier', styles['H3']))
    story.append(Paragraph(
        'Each token is classified on its own using the five features required by the assignment: '
        '<b>length</b> (MORE if the word has more than 4 characters, else LESS), the <b>word</b>, the '
        '<b>previous word</b>, the <b>next word</b> and the <b>previous tag</b>. Sentence boundaries are '
        'represented by &lt;s&gt; and &lt;/s&gt;. Features are assumed conditionally independent given the tag:',
        styles['Body']))
    story.append(Paragraph(
        't* = argmax_t [ log P(t) + &Sigma;_k log P(f_k | t) ]<br/>'
        f'P(f_k = v | t) = (C_k(t, v) + &alpha;) / (C(t) + &alpha;(|V_k| + 1)),   &alpha; = {hp["nb_laplace_alpha"]} (tuned on dev)',
        styles['Formula']))
    story.append(Paragraph(
        'At test time the gold previous tag is unavailable, so the sentence is decoded <b>greedily from left to '
        'right</b> and the previous-tag feature takes the classifier\'s own prediction for the preceding token.',
        styles['Body']))

    story.append(Paragraph('2.2 HMM parameter estimation', styles['H3']))
    story.append(Paragraph(
        'Each sentence is wrapped as &lt;s&gt; t<sub>1</sub> … t<sub>n</sub> &lt;/s&gt; and tag-bigram counts '
        'C(t<sub>i</sub>, t<sub>j</sub>) are collected. Transitions are add-k smoothed. Emissions use relative '
        'frequencies, and each tag keeps back a share of probability for unseen words. That share is '
        'proportional to H<sub>t</sub>, the number of hapax-legomena word types (types seen exactly once) '
        'carrying tag t. This is a Good-Turing-style estimate, so open classes such as N_NNP and JJ receive far '
        'more unknown-word mass than closed classes such as PSP.', styles['Body']))
    story.append(Paragraph(
        f'A_f(i&rarr;j) = P(t_k=j | t_(k-1)=i) = (C(i,j) + k) / (C(i) + k(T+1)),     k = {hp["hmm_transition_add_k"]} (tuned on dev)<br/>'
        'A_b(i&rarr;j) = P(t_k=j | t_(k+1)=i) = (C(j,i) + k) / (C(i) + k(T+1))<br/>'
        'P(w | t) = C(t, w) / (C(t) + H_t)      P(UNK | t) = H_t / (C(t) + H_t)', styles['Formula']))

    story.append(Paragraph('2.3 Forward (standard) Viterbi', styles['H3']))
    story.append(Paragraph(
        '&delta;_1(j) = log A_f(&lt;s&gt;&rarr;j) + log P(w_1|j)<br/>'
        '&delta;_k(j) = max_i [ &delta;_(k-1)(i) + log A_f(i&rarr;j) ] + log P(w_k|j),   &psi;_k(j) = argmax_i [...]<br/>'
        't_n = argmax_j [ &delta;_n(j) + log A_f(j&rarr;&lt;/s&gt;) ], then follow &psi; backwards to t_1.',
        styles['Formula']))
    story.append(Paragraph('2.4 Backward Viterbi', styles['H3']))
    story.append(Paragraph(
        'The recursion runs from right to left over the reversed chain. It starts at &lt;/s&gt;, uses the '
        'backward transitions A<sub>b</sub>, ends at &lt;s&gt;, and then follows the pointers forwards:',
        styles['Body']))
    story.append(Paragraph(
        '&delta;<sup>b</sup>_n(j) = log A_b(&lt;/s&gt;&rarr;j) + log P(w_n|j)<br/>'
        '&delta;<sup>b</sup>_k(j) = max_i [ &delta;<sup>b</sup>_(k+1)(i) + log A_b(i&rarr;j) ] + log P(w_k|j)<br/>'
        't_1 = argmax_j [ &delta;<sup>b</sup>_1(j) + log A_b(j&rarr;&lt;s&gt;) ], then follow the pointers forwards to t_n.',
        styles['Formula']))
    story.append(Paragraph('2.5 Forward-backward Viterbi', styles['H3']))
    story.append(Paragraph(
        'A full forward pass (&alpha;) and a full backward pass (&beta;) are run over the same trellis, and every '
        'position is labelled with the tag of highest posterior probability. Viterbi returns the single most '
        'probable <i>sequence</i>; this decoder instead maximises the expected number of correctly tagged '
        '<i>tokens</i>, which is what per-tag F1 and accuracy measure.', styles['Body']))
    story.append(Paragraph(
        '&alpha;_k(j) = P(w_1..w_k, t_k=j) = &Sigma;_i &alpha;_(k-1)(i) A_f(i&rarr;j) P(w_k|j)<br/>'
        '&beta;_k(i) = P(w_(k+1)..w_n | t_k=i) = &Sigma;_j A_f(i&rarr;j) P(w_(k+1)|j) &beta;_(k+1)(j)<br/>'
        't_k* = argmax_j &alpha;_k(j) &middot; &beta;_k(j)   ( &prop; P(t_k = j | w_1..w_n) )', styles['Formula']))
    story.append(Paragraph(
        '<b>Correctness check.</b> Replacing the sums with max (max-product forward &times; backward trellises) '
        'must reproduce the forward-Viterbi path. On both splits it agrees on '
        f'{pct(r["analysis"]["test"]["max_marginal_check"]["agreement_with_forward_viterbi"])}% of tokens.',
        styles['Body']))

    # --- 3. Results ---
    story.append(Paragraph('3. Results', styles['H2']))
    story.append(Paragraph('3.1 Overall performance', styles['H3']))
    rows = [['Algorithm', 'Split', 'Accuracy', 'Macro F1', 'Weighted F1', 'Known-word acc.', 'OOV-word acc.']]
    for a in ALGORITHMS:
        for s in ['dev', 'test']:
            m = splits[s][a]
            rows.append([SHORT[a] if s == 'dev' else '', s.upper(), pct(m['accuracy']), pct(m['macro_f1']),
                         pct(m['weighted_f1']), pct(m['known_accuracy']), pct(m['oov_accuracy'])])
    t = Table(rows, colWidths=[3.2 * cm, 1.3 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.7 * cm, 2.6 * cm])
    style = base_table_style(len(rows))
    for i in range(1, len(rows)):
        style.append(('BACKGROUND', (0, i), (-1, i), GREY_BG if ((i - 1) // 2) % 2 else colors.white))
    for s_off, s in enumerate(['dev', 'test']):
        for col, key in [(2, 'accuracy'), (3, 'macro_f1'), (4, 'weighted_f1'), (5, 'known_accuracy'), (6, 'oov_accuracy')]:
            b = max(range(len(ALGORITHMS)), key=lambda i: splits[s][ALGORITHMS[i]][key])
            style.append(('BACKGROUND', (col, 1 + 2 * b + s_off), (col, 1 + 2 * b + s_off), BEST_BG))
            style.append(('FONTNAME', (col, 1 + 2 * b + s_off), (col, 1 + 2 * b + s_off), 'Helvetica-Bold'))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Paragraph('Table 2: Overall scores (%). Green/bold marks the best value per split and metric. '
                           'OOV = word not seen in training.', styles['Caption']))

    story.append(Paragraph('3.2 Per-tag F1-score', styles['H3']))
    abbrev = ['NB', 'Fwd', 'Bwd', 'Fwd-Bwd']
    rows = [['', '', '', 'DEV F1', '', '', '', 'TEST F1', '', '', ''],
            ['Tag', 'Meaning', 'Test n'] + abbrev + abbrev]
    for tg in tags:
        row = [tg, TAG_NAMES.get(tg, ''), str(test[ALGORITHMS[0]]['per_tag'][tg]['support'])]
        row += [f'{dev[a]["per_tag"][tg]["f1"]:.3f}' for a in ALGORITHMS]
        row += [f'{test[a]["per_tag"][tg]["f1"]:.3f}' for a in ALGORITHMS]
        rows.append(row)
    t = Table(rows, colWidths=[1.65 * cm, 2.75 * cm, 1.05 * cm] + [1.45 * cm] * 8, repeatRows=2)
    style = base_table_style(len(rows)) + [('FONTSIZE', (0, 2), (-1, -1), 7)]
    style += [('SPAN', (3, 0), (6, 0)), ('SPAN', (7, 0), (10, 0)), ('BACKGROUND', (0, 1), (-1, 1), NAVY),
              ('TEXTCOLOR', (0, 1), (-1, 1), colors.white), ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
              ('FONTSIZE', (0, 0), (-1, 1), 7.5)]
    for i, tg in enumerate(tags, start=2):
        for base, sp in [(3, dev), (7, test)]:
            vals = [sp[a]['per_tag'][tg]['f1'] for a in ALGORITHMS]
            top = max(vals)
            for j, v in enumerate(vals):
                if v == top and top > 0:
                    style.append(('FONTNAME', (base + j, i), (base + j, i), 'Helvetica-Bold'))
                    style.append(('BACKGROUND', (base + j, i), (base + j, i), BEST_BG))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Paragraph('Table 3: Per-tag F1 on dev and test, tags ordered by training frequency. NB = Naive '
                           'Bayes; Fwd, Bwd, Fwd-Bwd = HMM forward, backward and forward-backward Viterbi. The best '
                           'algorithm for each tag and split is highlighted (ties are all highlighted).',
                           styles['Caption']))

    story.append(PageBreak())
    story.append(Paragraph('4. Charts', styles['H2']))
    story.append(figure('per_tag_f1_test.png', 'Figure 1: Per-tag F1 on the test split for all four algorithms.'))
    story.append(figure('per_tag_f1_dev.png', 'Figure 2: Per-tag F1 on the dev split for all four algorithms.'))
    story.append(figure('overall_comparison.png', 'Figure 3: Accuracy, macro-F1, weighted-F1 and OOV-word accuracy on dev and test.'))
    story.append(figure('f1_heatmap_test.png', 'Figure 4: Per-tag F1 heat map (test). Darker = higher F1.', width=11.5))
    story.append(figure('confusion_test.png', 'Figure 5: Row-normalised confusion matrices on test: Naive Bayes (left) vs the best HMM decoder (right).'))
    story.append(figure('tuning_dev.png', 'Figure 6: Dev-set sweeps used to choose the smoothing constants. In the HMM panel the backward-Viterbi curve sits exactly on top of the forward-Viterbi curve.'))

    # --- 5. Analysis ---
    story.append(Paragraph('5. Analysis and Discussion', styles['H2']))
    nb, fv, fb = test['naive_bayes'], test['forward_viterbi'], test['forward_backward_viterbi']
    gold_prev = r['analysis']['test']['naive_bayes_gold_prev_tag']['accuracy']
    tun = r['tuning_dev']['nb_alpha']
    story.append(Paragraph(
        f'<b>5.1 HMM vs Naive Bayes.</b> The HMM taggers beat Naive Bayes on test by '
        f'{(fv["accuracy"] - nb["accuracy"]) * 100:.2f} accuracy points ({pct(fv["accuracy"])}% vs '
        f'{pct(nb["accuracy"])}%) and by {(fv["macro_f1"] - nb["macro_f1"]) * 100:.1f} macro-F1 points. The gap '
        'is largest on low-frequency closed-class tags: RB, PR_PRL, QT_QTO, PR_PRQ, RP_NEG and N_NST. Naive '
        'Bayes has high <i>recall</i> but poor <i>precision</i> on these tags (for example N_NST precision '
        f'{nb["per_tag"]["N_NST"]["precision"]:.2f}, PR_PRL {nb["per_tag"]["PR_PRL"]["precision"]:.2f}, '
        f'RP_NEG {nb["per_tag"]["RP_NEG"]["precision"]:.2f}). The cause is Laplace smoothing on the open-vocabulary '
        'context features. An unseen previous/next word costs log(&alpha; / (C(t) + &alpha;|V|)), and this penalty '
        'is <i>smaller</i> for a rare tag (small C(t)) than for a frequent one. When the context words are new, '
        'rare tags are therefore pulled in. The HMM conditions only on the neighbouring <i>tags</i> (24 values, '
        'not tens of thousands of words) and scores the whole sentence jointly, so it avoids this bias.',
        styles['Body']))
    story.append(Paragraph(
        f'<b>5.2 Effect of smoothing and of the previous-tag feature in Naive Bayes.</b> Plain add-one '
        f'(&alpha; = 1) gives only {pct(tun["1.0"])}% dev accuracy, while the tuned &alpha; = {hp["nb_laplace_alpha"]} '
        f'gives {pct(tun[str(hp["nb_laplace_alpha"])])}%. Add-one gives too much mass to unseen word values, which '
        'is the same over-smoothing effect seen in the n-gram labs. Supplying the <i>gold</i> previous tag instead '
        f'of the predicted one raises test accuracy only from {pct(nb["accuracy"])}% to {pct(gold_prev)}%, so '
        'greedy error propagation costs about '
        f'{(gold_prev - nb["accuracy"]) * 100:.1f} points. Most of the gap to the HMM comes from the independence '
        'assumption itself rather than from that feature.', styles['Body']))
    story.append(Paragraph(
        '<b>5.3 Why forward and backward Viterbi give identical output.</b> The two decoders produced exactly the '
        'same tag for every token on both splits, and for every k in the sweep. This is expected, not a bug. On '
        'the augmented chain &lt;s&gt; t<sub>1</sub>…t<sub>n</sub> &lt;/s&gt; every tag occurrence has exactly one '
        'predecessor and one successor, so the row sum and column sum of the bigram matrix both equal C(t), and '
        'both boundary symbols occur once per sentence. Hence:', styles['Body']))
    story.append(Paragraph(
        '&Pi;_k A_f(t_(k-1)&rarr;t_k) = &Pi;_k (C(t_(k-1),t_k)+k) / &Pi;_k (C(t_(k-1))+k(T+1))<br/>'
        '&Pi;_k A_b(t_(k+1)&rarr;t_k) = &Pi;_k (C(t_k,t_(k+1))+k) / &Pi;_k (C(t_(k+1))+k(T+1))<br/>'
        'The numerators run over the same bigrams, and the denominators run over the same multiset {&lt;s&gt;/&lt;/s&gt;, t_1, …, t_n}.',
        styles['Formula']))
    story.append(Paragraph(
        'The left-to-right and right-to-left HMMs therefore define <b>the same joint distribution</b> '
        'P(tags, words), so their argmax sequences coincide. Numerically, the largest absolute difference in '
        f'log P(tags, words) over all dev sentences is {r["forward_backward_joint_max_abs_logprob_gap"]:.1e}, which '
        'is floating-point rounding. Running Viterbi in the other direction changes only the order of the '
        'dynamic program, not the model, so it cannot change the answer.', styles['Body']))
    story.append(Paragraph(
        f'<b>5.4 Forward-backward (posterior) decoding.</b> Posterior decoding changed '
        f'{(1 - r["analysis"]["test"]["pairwise_agreement"]["forward_viterbi|forward_backward_viterbi"]) * 100:.2f}% '
        f'of test tokens relative to Viterbi. Accuracy moved by '
        f'{(fb["accuracy"] - fv["accuracy"]) * 100:+.2f} points on test and '
        f'{(dev["forward_backward_viterbi"]["accuracy"] - dev["forward_viterbi"]["accuracy"]) * 100:+.2f} on dev, and '
        f'OOV accuracy moved by {(fb["oov_accuracy"] - fv["oov_accuracy"]) * 100:+.2f} points on test. Posterior '
        'decoding is optimal for expected token accuracy <i>under the model</i>. With a well-peaked bigram HMM '
        'the per-token posteriors usually agree with the best path, so in practice the two decoders are '
        'statistically tied. Its gains are concentrated in ambiguous regions such as runs of unknown words.',
        styles['Body']))
    conf = fv['top_confusions'][:4]
    conf_txt = ', '.join(f'{c["gold"]}&rarr;{c["pred"]} ({c["count"]})' for c in conf)
    story.append(Paragraph(
        f'<b>5.5 Error analysis.</b> The most frequent HMM errors on test are {conf_txt}. Separating common from '
        'proper nouns is the hardest decision in Hindi: there is no capitalisation, and many names (person, '
        'place and organisation names) are also ordinary words. The noun/adjective boundary is also fuzzy. '
        f'Unknown words are the main weakness of every model: OOV accuracy is only {pct(fv["oov_accuracy"])}% '
        f'(HMM) and {pct(nb["oov_accuracy"])}% (NB), against about {pct(fv["known_accuracy"])}% for known words, '
        'because the only signal for an unseen word is its context. Suffix-based emission models for unknown '
        'words (Hindi inflectional endings for verbs, plurals and adjectives) would be the most effective '
        'next step.', styles['Body']))
    story.append(Paragraph(
        f'<b>5.6 A note on macro-F1.</b> Test macro-F1 for the HMM ({pct(fv["macro_f1"])}%) is lower than on dev '
        f'({pct(dev["forward_viterbi"]["macro_f1"])}%) mainly because the test split contains a single RP_INJ '
        '(interjection) token, a tag with only 11 training occurrences. Missing that one token gives the tag '
        'an F1 of 0 and pulls the 24-tag average down by about 4 points. Weighted F1 and accuracy, which reflect '
        'token frequency, are essentially identical on dev and test.', styles['Body']))

    story.append(Paragraph('6. Conclusion', styles['H2']))
    story.append(Paragraph(
        f'A bigram HMM decoded with Viterbi tags Hindi text with {pct(fv["accuracy"])}% test accuracy, '
        f'{(fv["accuracy"] - nb["accuracy"]) * 100:.1f} points above a carefully smoothed Naive Bayes classifier '
        'that uses word, context-word, length and previous-tag features. Modelling the tag <i>sequence</i> '
        'jointly matters most for rare closed-class tags. Backward Viterbi is mathematically equivalent to '
        'forward Viterbi for a bigram HMM estimated from counts, and the experiments confirm this exactly. '
        'Forward-backward posterior decoding is statistically tied with Viterbi. The remaining errors are '
        'dominated by the N_NN/N_NNP distinction and by out-of-vocabulary words.', styles['Body']))

    doc.build(story)
    print(f'Wrote {REPORT_PATH}')


if __name__ == '__main__':
    build()
