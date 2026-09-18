"""Builds LAB-5-Report.pdf: a formatted lab report with color-coded result
tables and the comparison charts from output/results.json + assets/*.png.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, HRFlowable,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'
REPORT_PATH = ROOT / 'LAB-5-Report.pdf'

NAVY = colors.HexColor('#2C3E50')
BLUE = colors.HexColor('#4C72B0')
GREEN = colors.HexColor('#55A868')
GREEN_BG = colors.HexColor('#E7F4EA')
RED = colors.HexColor('#C44E52')
RED_BG = colors.HexColor('#FBEAEA')
GREY_BG = colors.HexColor('#F4F5F7')
ORDER_NAMES = ['unigram', 'bigram', 'trigram', 'quadrigram']

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('ReportTitle', parent=styles['Title'], fontSize=22, textColor=NAVY, spaceAfter=6))
styles.add(ParagraphStyle('ReportSubtitle', parent=styles['Normal'], fontSize=12, textColor=colors.HexColor('#555'), alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle('H2', parent=styles['Heading2'], textColor=NAVY, spaceBefore=16, spaceAfter=8))
styles.add(ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10.2, leading=14.5, spaceAfter=8))
styles.add(ParagraphStyle('Formula', parent=styles['Code'], fontSize=10.5, alignment=TA_CENTER, textColor=NAVY, backColor=GREY_BG, borderPadding=8, spaceAfter=10))
styles.add(ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8.5, textColor=colors.HexColor('#666'), alignment=TA_CENTER, spaceBefore=4, spaceAfter=12))


def stat_card(value, label, color):
    t = Table([[Paragraph(f'<font color="{color.hexval()}" size="18"><b>{value}</b></font>')],
               [Paragraph(f'<font size="8.5" color="#555">{label}</font>')]],
              colWidths=[4.2 * cm])
    t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#DDD')),
        ('BACKGROUND', (0, 0), (-1, -1), GREY_BG),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 10),
    ]))
    return t


def build():
    results = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    k = results['k']
    doc = SimpleDocTemplate(str(REPORT_PATH), pagesize=A4,
                             topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                             leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    story = []

    # --- Title page ---
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph('NLP Laboratory — Assignment 5', styles['ReportTitle']))
    story.append(Paragraph(f'Add-k Smoothing (k = {k}) for N-gram Language Models', styles['ReportSubtitle']))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#DDD')))
    story.append(Spacer(1, 0.6 * cm))

    meta = Table([
        ['Name', '_' * 30, 'Course', 'Natural Language Processing'],
        ['Roll No.', '_' * 30, 'Lab', 'Lab 5 — Add-k Smoothing'],
    ], colWidths=[2.3 * cm, 5.5 * cm, 2 * cm, 6.4 * cm])
    meta.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('TEXTCOLOR', (0, 0), (0, -1), NAVY),
        ('TEXTCOLOR', (2, 0), (2, -1), NAVY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta)
    story.append(Spacer(1, 0.8 * cm))

    cards = Table([[
        stat_card(f'{results["vocab_size"]:,}', 'Vocabulary size (V)', BLUE),
        stat_card(k, 'Smoothing constant k', GREEN),
        stat_card('4', 'N-gram orders evaluated', BLUE),
        stat_card(f'{results["test_oov_rate"]*100:.2f}%', 'Test OOV rate', RED),
    ]], colWidths=[4.4 * cm] * 4)
    cards.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story.append(cards)
    story.append(Spacer(1, 0.8 * cm))

    story.append(Paragraph(
        'This lab reuses the four n-gram language models (unigram, bigram, trigram, '
        'quadrigram) trained on the Gujarati corpus in Assignment 4, and rescores them '
        'with <b>Add-k smoothing</b> instead of Add-one (Laplace) smoothing, on the '
        'same held-out dev and test splits.', styles['Body']))
    story.append(PageBreak())

    # --- Overview / formula ---
    story.append(Paragraph('1. Overview', styles['H2']))
    story.append(Paragraph(
        'Add-k smoothing generalizes Laplace (add-one) smoothing by replacing the '
        'fixed "+1" with a tunable constant k, redistributing less probability mass '
        'to unseen n-grams than Laplace does. All counts (n-gram counts, context '
        'counts, vocabulary, and the &lt;UNK&gt; mapping) are identical to Assignment 4 — '
        'only the scoring formula changes — so the two labs\' perplexities are directly '
        'comparable.', styles['Body']))
    story.append(Paragraph(
        'P_add-k( w_i | w_i-N+1 ... w_i-1 )&nbsp;&nbsp;=&nbsp;&nbsp;'
        '[ C(w_i-N+1 ... w_i) + k ]&nbsp;&nbsp;/&nbsp;&nbsp;'
        '[ C(w_i-N+1 ... w_i-1) + k&middot;V ]',
        styles['Formula']))
    story.append(Paragraph(
        f'with <b>k = {k}</b> and V = {results["vocab_size"]:,} (kept training words + '
        '&lt;UNK&gt; + &lt;/s&gt;, singleton words collapsed to &lt;UNK&gt;).', styles['Body']))

    # --- Results table ---
    story.append(Paragraph('2. Results', styles['H2']))
    header = ['Order', 'Laplace\ndev PP', 'Add-k\ndev PP', 'Laplace\ntest PP', 'Add-k\ntest PP', 'Test\nimprovement']
    rows = [header]
    row_colors = []
    for o in ORDER_NAMES:
        p = results['perplexity'][o]
        pct = 100 * (p['laplace_test_perplexity'] - p['add_k_test_perplexity']) / p['laplace_test_perplexity']
        rows.append([
            o.capitalize(),
            f'{p["laplace_dev_perplexity"]:,.1f}',
            f'{p["add_k_dev_perplexity"]:,.1f}',
            f'{p["laplace_test_perplexity"]:,.1f}',
            f'{p["add_k_test_perplexity"]:,.1f}',
            f'+{pct:.1f}%',
        ])
        row_colors.append(GREEN_BG if pct > 0 else RED_BG)

    table = Table(rows, colWidths=[2.6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.7 * cm])
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (5, 1), (5, -1), GREEN),
        ('FONTNAME', (5, 1), (5, -1), 'Helvetica-Bold'),
    ]
    for i, c in enumerate(row_colors, start=1):
        style.append(('BACKGROUND', (0, i), (4, i), colors.white if i % 2 else GREY_BG))
        style.append(('BACKGROUND', (5, i), (5, i), c))
    table.setStyle(TableStyle(style))
    story.append(table)
    story.append(Paragraph(
        'Add-k (k=0.3) reduces perplexity relative to Laplace (k=1) at every order, on both '
        'dev and test — the green improvement column is positive throughout.', styles['Caption']))

    # --- Charts ---
    story.append(Paragraph('3. Comparison Charts', styles['H2']))
    for name, cap in [
        ('addk_vs_laplace_dev.png', 'Figure 1 — Dev-set perplexity, Add-k vs. Laplace, by n-gram order.'),
        ('addk_vs_laplace_test.png', 'Figure 2 — Test-set perplexity, Add-k vs. Laplace, by n-gram order.'),
        ('pct_improvement.png', 'Figure 3 — Percent perplexity reduction of Add-k over Laplace, by order.'),
    ]:
        img_path = ASSETS / name
        img = Image(str(img_path), width=15 * cm, height=15 * cm * 0.6)
        story.append(img)
        story.append(Paragraph(cap, styles['Caption']))

    story.append(PageBreak())

    # --- Analysis ---
    story.append(Paragraph('4. Analysis', styles['H2']))
    story.append(Paragraph(
        'The size of the improvement is a story about the smoothing constant relative to V, '
        'not about k in isolation. The correction term subtracted from every seen n-gram\'s '
        'share of probability mass scales with k&middot;V; since V = 222,199, Laplace\'s k=1 '
        'term dwarfs typical observed counts from the bigram order upward, so most probability '
        'mass ends up redistributed onto n-grams that were never seen. Shrinking k to 0.3 cuts '
        'that term to 30% of Laplace\'s, which is why the largest relative gain (about 44%) '
        'appears at the <b>bigram</b> order, where counts are populated enough for the smoothing '
        'constant to be the dominant source of distortion.', styles['Body']))
    story.append(Paragraph(
        'The <b>unigram</b> order barely changes (&lt;0.5%) because there is no context to '
        'sparsify — the denominator is just the total token count (~16.2M), against which '
        'even Laplace\'s k&middot;V &asymp; 222,199 is negligible. The <b>trigram</b> and '
        '<b>quadrigram</b> orders improve less than the bigram in relative terms because at '
        'those orders almost every test n-gram is itself unseen in training regardless of k — '
        'both models remain dominated by the same "probability mass spread across an '
        'astronomically large unseen n-gram space" failure mode Assignment 4 diagnosed for '
        'Laplace smoothing. Add-k with k=0.3 is a smaller version of that over-smoothing, not '
        'a fix for it: perplexity still climbs with order under both schemes, since neither '
        'backs off to a lower-order estimate for n-grams it has never seen.', styles['Body']))
    story.append(Paragraph(
        'A method like Good-Turing, Kneser-Ney, or interpolation/backoff — which redistribute '
        'mass based on how *often* low-count events occur rather than a flat additive constant — '
        'would be expected to improve, rather than degrade, as order increases.', styles['Body']))

    story.append(Paragraph('5. Conclusion', styles['H2']))
    story.append(Paragraph(
        f'Add-k smoothing with k={k} consistently outperforms Add-one (Laplace) smoothing on '
        'this Gujarati corpus across all four n-gram orders and both held-out splits, '
        'confirming that the smoothing constant is a meaningful hyperparameter rather than an '
        'arbitrary implementation detail. The improvement is largest where counts are dense '
        'enough for the constant to dominate the estimate (bigram) and smallest where it '
        'barely matters (unigram) or where sparsity overwhelms it regardless of k '
        '(quadrigram).', styles['Body']))

    doc.build(story)
    print(f'Wrote {REPORT_PATH}')


if __name__ == '__main__':
    build()
