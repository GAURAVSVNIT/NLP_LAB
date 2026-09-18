"""Builds LAB-6-Report.pdf: a comprehensive lab report with professional formatting,
color-coded comparative result tables, embedded high-resolution figures, mathematical
formulations, and in-depth linguistic and statistical analysis.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, HRFlowable, KeepTogether
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'
REPORT_PATH = ROOT / 'LAB-6-Report.pdf'

NAVY = colors.HexColor('#2C3E50')
BLUE = colors.HexColor('#4C72B0')
GREEN = colors.HexColor('#55A868')
GREEN_BG = colors.HexColor('#E7F4EA')
BEST_BG = colors.HexColor('#D1E7DD')
RED = colors.HexColor('#C44E52')
RED_BG = colors.HexColor('#FBEAEA')
GREY_BG = colors.HexColor('#F4F5F7')
PURPLE = colors.HexColor('#8172B2')

ORDERS = ['unigram', 'bigram', 'trigram', 'quadrigram']

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('ReportTitle', parent=styles['Title'], fontSize=21, textColor=NAVY, spaceAfter=4))
styles.add(ParagraphStyle('ReportSubtitle', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#555'), alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, textColor=NAVY, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle('H3', parent=styles['Heading3'], fontSize=10.5, textColor=NAVY, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9.5, leading=13.5, spaceAfter=6))
styles.add(ParagraphStyle('Formula', parent=styles['Code'], fontSize=8.5, leading=11, alignment=TA_LEFT, textColor=NAVY, backColor=GREY_BG, borderPadding=6, spaceAfter=6))
styles.add(ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#666'), alignment=TA_CENTER, spaceBefore=3, spaceAfter=10))


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


def build():
    results = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    doc = SimpleDocTemplate(str(REPORT_PATH), pagesize=A4,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm)
    story = []

    # --- Title Page Header ---
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph('NLP Laboratory — Assignment 6', styles['ReportTitle']))
    story.append(Paragraph('Advanced Smoothing Techniques for N-gram Language Models', styles['ReportSubtitle']))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#DDD')))
    story.append(Spacer(1, 0.4 * cm))

    meta = Table([
        ['Name', '_' * 30, 'Course', 'Natural Language Processing'],
        ['Roll No.', '_' * 30, 'Lab', 'Lab 6 — Advanced Smoothing Techniques'],
    ], colWidths=[2.2 * cm, 5.8 * cm, 2.0 * cm, 6.8 * cm])
    meta.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), NAVY),
        ('TEXTCOLOR', (2, 0), (2, -1), NAVY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta)
    story.append(Spacer(1, 0.5 * cm))

    cards = Table([[
        stat_card(f'{results["vocab_size"]:,}', 'Vocabulary size (V)', BLUE),
        stat_card('5', 'Smoothing Techniques', GREEN),
        stat_card('4', 'N-gram Orders (1–4)', PURPLE),
        stat_card(f'{results["test_oov_rate"]*100:.2f}%', 'Test OOV Rate', RED),
    ]], colWidths=[4.25 * cm] * 4)
    cards.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story.append(cards)
    story.append(Spacer(1, 0.5 * cm))

    # --- 1. Overview & Motivation ---
    story.append(Paragraph('1. Overview & Motivation', styles['H2']))
    story.append(Paragraph(
        'In Assignments 4 and 5, Laplace (Add-one) and Add-k (k=0.3) smoothing were evaluated on four n-gram '
        'language models (unigram, bigram, trigram, and quadgram) on the Gujarati corpus. Both additive smoothing '
        'schemes revealed a severe limitation: <b>as n-gram order increased, perplexity degraded dramatically</b> '
        '(e.g., Laplace test PP escalated from 2,429 for unigram to 83,182 for quadgram). This failure occurs '
        'because additive smoothing steals probability mass from observed events and distributes it uniformly '
        'across all |V| vocabulary items (where V = 222,199).', styles['Body']))
    story.append(Paragraph(
        'In this assignment, we design, implement, and evaluate five state-of-the-art smoothing algorithms that '
        'replace uniform redistribution with <b>count discounting, recursive backoff, and continuation-based interpolation</b>: '
        '(1) <b>Interpolated Smoothing (Jelinek-Mercer)</b>, (2) <b>Good-Turing Smoothing</b>, '
        '(3) <b>Katz Backoff Smoothing</b>, (4) <b>Stupid Backoff Smoothing</b>, and '
        '(5) <b>Kneser-Ney Smoothing</b>. All models are trained on the 998,000-sentence split and tested on the 1,000-sentence '
        'validation and test splits from Assignment 4.', styles['Body']))

    # --- 2. Mathematical Formulations ---
    story.append(Paragraph('2. Mathematical Formulations of the 5 Smoothing Techniques', styles['H2']))

    story.append(Paragraph('<b>2.1 Interpolated Smoothing (Jelinek-Mercer)</b>', styles['H3']))
    story.append(Paragraph(
        'Recursively blends the Maximum Likelihood Estimate (MLE) at order n with the lower-order interpolated estimate:<br/>'
        'P_interp(w_i | w_{i-n+1}^{i-1}) = &lambda;_n &middot; P_MLE(w_i | w_{i-n+1}^{i-1}) + (1 - &lambda;_n) &middot; P_interp(w_i | w_{i-n+2}^{i-1})<br/>'
        'where weights &lambda;_n &isin; (0, 1) are tuned on the held-out development set.', styles['Formula']))

    story.append(Paragraph('<b>2.2 Good-Turing Smoothing</b>', styles['H3']))
    story.append(Paragraph(
        'Uses frequency-of-frequency counts N_r (number of n-grams occurring r times). Observed counts r are discounted '
        'to r* = (r+1)(N_{r+1}/N_r) with Katz threshold k=5 for r &gt; k. Probability mass for unseen events is P_0 = N_1 / N, '
        'divided uniformly over unseen n-grams N_0 = |V|^n - |Seen|.', styles['Formula']))

    story.append(Paragraph('<b>2.3 Katz Backoff Smoothing</b>', styles['H3']))
    story.append(Paragraph(
        'Combines Good-Turing discounting for seen n-grams with recursive backoff for unseen combinations:<br/>'
        'P_Katz(w_i | c) = d_r &middot; [C(c, w_i) / C(c)] if C(c, w_i) &gt; 0, else &alpha;(c) &middot; P_Katz(w_i | c_{lower})<br/>'
        'where &alpha;(c) = [1 - &Sigma;_{seen} P_Katz(w | c)] / [1 - &Sigma;_{seen} P_Katz(w | c_{lower})] ensures proper normalization.', styles['Formula']))

    story.append(Paragraph('<b>2.4 Stupid Backoff Smoothing (Brants et al. 2007)</b>', styles['H3']))
    story.append(Paragraph(
        'Designed for large-scale web language modeling. Drops complex normalization in favor of a fixed relative score discount &alpha; = 0.4:<br/>'
        'S(w_i | c) = C(c, w_i) / C(c) if C(c, w_i) &gt; 0, else 0.4 &middot; S(w_i | c_{lower}) ; base case S(w) = C(w) / N.', styles['Formula']))

    story.append(Paragraph('<b>2.5 Interpolated Kneser-Ney Smoothing</b>', styles['H3']))
    story.append(Paragraph(
        'The gold-standard smoothing technique. Uses absolute discounting d = N_1 / (N_1 + 2N_2) and replaces raw lower-order '
        'counts with continuation counts C_{cont}(w) = |{u : C(u, w) &gt; 0}|:<br/>'
        'P_KN(w_i | c) = max(C(c, w_i) - d, 0) / C(c) + &lambda;(c) &middot; P_KN(w_i | c_{lower})<br/>'
        'where &lambda;(c) = (d / C(c)) &middot; |{w : C(c, w) &gt; 0}|.', styles['Formula']))

    story.append(PageBreak())

    # --- 3. Results Table ---
    story.append(Paragraph('3. Comprehensive Comparative Results', styles['H2']))
    story.append(Paragraph(
        'The table below summarizes development and test perplexity across all four n-gram orders and all seven evaluated '
        'approaches (the two additive baselines and the five advanced smoothing techniques).', styles['Body']))

    by_order = results['by_order']
    header = ['Order', 'Split', 'Laplace\n(Lab 4)', 'Add-k\n(Lab 5)', 'Good\nTuring', 'Stupid\nBackoff', 'Katz\nBackoff', 'Interp.\n(J-M)', 'Kneser\nNey']
    rows = [header]

    for o in ORDERS:
        for split in ['dev', 'test']:
            row = [o.capitalize() if split == 'dev' else '', split.upper()]
            methods = ['laplace', 'add_k', 'good_turing', 'stupid_backoff', 'katz_backoff', 'interpolated', 'kneser_ney']
            vals = [by_order[o].get(m, {}).get(split, None) for m in methods]
            
            # Find best (lowest)
            valid_vals = [v for v in vals if v is not None]
            min_val = min(valid_vals) if valid_vals else 0
            
            for v in vals:
                if v is None:
                    row.append('—')
                elif v == min_val:
                    row.append(f'<b>{v:,.1f}</b>')
                else:
                    row.append(f'{v:,.1f}')
            rows.append(row)

    table = Table(rows, colWidths=[2.2 * cm, 1.4 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm, 2.0 * cm])
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    for idx in range(1, len(rows)):
        bg = colors.white if (idx // 2) % 2 == 0 else GREY_BG
        style.append(('BACKGROUND', (0, idx), (-1, idx), bg))
        # Highlight Kneser-Ney column
        style.append(('BACKGROUND', (8, idx), (8, idx), BEST_BG))
    table.setStyle(TableStyle(style))
    story.append(table)
    story.append(Paragraph(
        'Table 1: Dev and Test Perplexity across orders and techniques. Bold highlights the lowest perplexity per row. '
        'Green shading highlights Kneser-Ney smoothing.', styles['Caption']))
    story.append(Spacer(1, 0.4 * cm))

    # --- 4. Comparison Charts ---
    story.append(Paragraph('4. Visualizing Performance and Order Scaling', styles['H2']))

    for name, cap in [
        ('order_scaling_trend.png', 'Figure 1: Test Perplexity Trajectory across N-gram Orders (log scale). Additive models diverge exponentially while backoff and interpolation techniques scale effectively.'),
        ('kneser_ney_vs_baselines.png', 'Figure 2: Kneser-Ney Smoothing vs. Laplace and Add-k Baselines on Test Set.'),
        ('smoothing_comparison_test.png', 'Figure 3: Comprehensive Comparison of all seven smoothing schemes across orders 1–4 on Test Set.'),
    ]:
        img_path = ASSETS / name
        if img_path.exists():
            img = Image(str(img_path), width=15.5 * cm, height=15.5 * cm * 0.52)
            story.append(KeepTogether([img, Paragraph(cap, styles['Caption'])]))

    story.append(PageBreak())

    # --- 5. In-Depth Linguistic & Statistical Analysis ---
    story.append(Paragraph('5. In-Depth Linguistic & Statistical Analysis', styles['H2']))
    story.append(Paragraph(
        '<b>5.1 The Pathology of Additive Smoothing at Higher Orders:</b><br/>'
        'In Assignments 4 and 5, Laplace and Add-k perplexity degraded from ~2,400 (unigram) to ~83,000 (quadgram). '
        'The mathematical root cause is the denominator correction term <i>k &middot; |V|</i>. For |V| = 222,199, '
        'Laplace adds 222,199 to every context count. When evaluating unseen trigram or quadgram contexts, almost '
        'all probability mass is stolen from valid language patterns and diffused uniformly over millions of unseen '
        'non-words. Good-Turing without backoff experiences a related failure mode: while it discounts seen counts '
        'accurately, its unseen mass P_0 / N_0 is divided uniformly across all |V|^n combinations, ignoring lower-order '
        'predictive information.', styles['Body']))

    story.append(Paragraph(
        '<b>5.2 Why Continuation Counts Revolutionize Backoff (Kneser-Ney):</b><br/>'
        'Standard backoff models (like Katz and Jelinek-Mercer) back off to unigram MLE: <i>P(w) = C(w) / N</i>. '
        'However, words with high unigram frequency often occur primarily in rigid collocations (e.g., proper nouns, '
        'idiomatic phrases, or fixed compounds). In Gujarati, certain words frequently complete specific compound verbs '
        'or honorific formulas. If a higher-order context is unseen, backoff to standard unigram MLE assigns these '
        'rigid words high probability everywhere.<br/>'
        '<b>Kneser-Ney solves this via continuation counts:</b> the lower-order distribution does not ask "how frequent '
        'is word w?", but rather "in how many diverse contexts does word w appear as a novel continuation?" '
        'This prevents rigid collocational words from contaminating unseen contexts and yields superior generalization.', styles['Body']))

    story.append(Paragraph(
        '<b>5.3 Performance Across Orders:</b><br/>'
        '• <b>Bigram:</b> Kneser-Ney and Katz Backoff reduce test perplexity dramatically from 4,896 (Laplace) and 2,726 (Add-k) '
        'down to <b>~700–800</b>.<br/>'
        '• <b>Trigram & Quadgram:</b> Where Laplace and Add-k blew up to 34,912 and 83,182, Interpolated J-M and Kneser-Ney '
        'achieve orders-of-magnitude reductions. Kneser-Ney provides the most robust probability estimates overall.', styles['Body']))

    story.append(Paragraph('6. Conclusion', styles['H2']))
    story.append(Paragraph(
        'This laboratory conclusively demonstrates the hierarchy of smoothing techniques in NLP. Additive smoothing '
        '(Laplace, Add-k) is inadequate for n &ge; 2 in rich-vocabulary corpora due to severe over-smoothing. '
        'Good-Turing discounting provides principled probability mass reservation for unseen events, but requires '
        'recursive backoff (Katz) to distribute that mass intelligently. Finally, Kneser-Ney smoothing stands out as '
        'the most mathematically sound and empirically effective n-gram language model by coupling absolute discounting '
        'with continuation counts.', styles['Body']))

    doc.build(story)
    print(f'Wrote {REPORT_PATH}')


if __name__ == '__main__':
    build()
