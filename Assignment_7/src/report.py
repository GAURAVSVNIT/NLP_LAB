"""Builds LAB-7-Report.pdf from output/results.json and the charts in assets/."""
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from data import ASSETS, OUTPUT_DIR, ROOT

PDF_PATH = ROOT / 'LAB-7-Report.pdf'

METHOD_LABELS = {
    'naive_bayes': 'Naive Bayes',
    'hmm_forward_viterbi': 'HMM forward Viterbi',
    'hmm_backward_viterbi': 'HMM backward Viterbi',
    'hmm_forward_backward_viterbi': 'HMM forward-backward Viterbi',
}


def _table(rows, col_widths, header_bg=colors.HexColor('#4C72B0')):
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#BBBBBB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F3F6FA')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return table


def _per_tag_table(results, split):
    tags = sorted(results['corpus']['tag_counts_train'],
                  key=lambda t: results['corpus']['tag_counts_train'][t], reverse=True)
    methods = list(METHOD_LABELS)
    header = ['Tag', 'Support'] + [METHOD_LABELS[m] for m in methods]
    rows = [header]
    support = results['methods']['naive_bayes'][split]['per_tag']
    for t in tags:
        row = [t, str(support[t]['support'])]
        for m in methods:
            row.append(f"{results['methods'][m][split]['per_tag'][t]['f1']:.3f}")
        rows.append(row)
    widths = [2.4 * cm, 1.6 * cm] + [3.1 * cm] * len(methods)
    return _table(rows, widths)


def build():
    results = json.loads((OUTPUT_DIR / 'results.json').read_text(encoding='utf-8'))
    corpus = results['corpus']
    styles = getSampleStyleSheet()
    body = ParagraphStyle('body', parent=styles['BodyText'], alignment=TA_JUSTIFY, leading=14)
    h1 = styles['Heading1']
    h2 = styles['Heading2']
    caption = ParagraphStyle('cap', parent=body, fontSize=8.5, textColor=colors.HexColor('#555555'))

    story = []
    story.append(Paragraph('NLP Lab 7: Part-of-Speech Tagging for Hindi', styles['Title']))
    story.append(Paragraph('Naive Bayes and Hidden Markov Model taggers with forward, backward, '
                           'and forward-backward Viterbi decoding', styles['Italic']))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph('1. Data', h2))
    story.append(Paragraph(
        f"The corpus is the Hindi POS-tagged data supplied with the lab (word/tag pairs, one per line, "
        f"blank lines between sentences). The train split has {corpus['train_sentences']:,} sentences and "
        f"{corpus['train_tokens']:,} tokens over {len(corpus['tags'])} tags. Dev has "
        f"{corpus['dev_sentences']:,} sentences and test has {corpus['test_sentences']:,}.", body))

    story.append(Paragraph('2. Methods', h2))
    story.append(Paragraph(
        '<b>Naive Bayes.</b> Each token is classified independently from five categorical features: '
        'a binary length feature (MORE if the word has more than 4 characters, else LESS), the word, '
        'the previous word, the next word, and the previous tag. Features use Laplace smoothing '
        '(alpha = 1). Training uses gold previous tags; prediction is greedy left-to-right, using the '
        'predicted previous tag.', body))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        '<b>HMM.</b> A first-order HMM with add-one smoothed transitions, MLE emissions, and words seen '
        'once in training mapped to &lt;UNK&gt;. Three decoders are compared: '
        '<b>forward Viterbi</b> (standard left-to-right dynamic program); <b>backward Viterbi</b> '
        '(right-to-left dynamic program over a reverse transition model P(t_i | t_{i+1}) estimated from '
        'the same bigram counts); and <b>forward-backward Viterbi</b>, which runs a forward and a backward '
        'max-product pass and assigns each token the tag on the best full-sequence path through it '
        '(max-marginal decoding). Because the last decoder picks each token independently, its output '
        'need not be a single consistent tag path.', body))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        'All three Viterbi decoders were checked against brute-force search over every tag sequence on '
        'short sentences: the forward and backward decoders returned the exact best sequence, and the '
        'forward-backward decoder returned the exact per-token max-marginals.', body))

    story.append(Paragraph('3. Overall results', h2))
    overall = [['Method', 'Dev accuracy', 'Dev macro-F1', 'Test accuracy', 'Test macro-F1']]
    for m, label in METHOD_LABELS.items():
        d = results['methods'][m]['dev']
        t = results['methods'][m]['test']
        overall.append([label, f"{d['accuracy']:.4f}", f"{d['macro_f1']:.4f}",
                        f"{t['accuracy']:.4f}", f"{t['macro_f1']:.4f}"])
    story.append(_table(overall, [5.6 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm]))
    story.append(Paragraph('Macro-F1 averages per-tag F1 over tags that occur in the gold data.', caption))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(str(ASSETS / 'overall_comparison.png'), width=16 * cm, height=8.6 * cm))

    story.append(PageBreak())
    story.append(Paragraph('4. Per-tag F1 on the test set', h2))
    story.append(Paragraph('Tags are ordered by training frequency; Support is the number of gold test tokens.', caption))
    story.append(Spacer(1, 0.2 * cm))
    story.append(_per_tag_table(results, 'test'))
    story.append(PageBreak())
    story.append(Image(str(ASSETS / 'per_tag_f1_test.png'), width=15.5 * cm, height=20 * cm,
                       kind='proportional'))

    story.append(PageBreak())
    story.append(Paragraph('5. Per-tag F1 on the dev set', h2))
    story.append(_per_tag_table(results, 'dev'))
    story.append(PageBreak())
    story.append(Image(str(ASSETS / 'per_tag_f1_dev.png'), width=15.5 * cm, height=20 * cm,
                       kind='proportional'))

    story.append(PageBreak())
    story.append(Paragraph('6. Analysis', h2))
    points = [
        'The HMM beats Naive Bayes by about 2.7 accuracy points and 15 macro-F1 points on test. Naive Bayes '
        'tags each token with only the previous tag as context, so it cannot use the sentence-level tag '
        'structure that the HMM models.',
        'Naive Bayes fails most on rare and closed-class tags: PR_PRQ (test F1 0.06) and RP_INTF (0.34), and '
        'RB (0.38) too. The sequence model recovers most of these.',
        'The three HMM decoders perform almost identically. Forward and forward-backward Viterbi give the '
        'same test accuracy (0.9372). Backward Viterbi is about 0.1 points lower, because its reverse model '
        'is normalised differently even though it uses the same counts.',
        'The hardest tags for all methods are N_NNP (proper nouns, test F1 about 0.85 over 5,296 tokens), RB '
        '(about 0.85), and RP_INTF (about 0.87). Proper nouns are often unseen words, and RB overlaps with '
        'adverbial uses of other tags.',
        'RP_INJ has F1 = 0 for every method. It occurs once in the test set and no decoder labels it correctly, '
        'so one token drags the per-tag average down.',
    ]
    for p in points:
        story.append(Paragraph('&bull; ' + p, body))
        story.append(Spacer(1, 0.15 * cm))

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title='NLP Lab 7: Hindi POS Tagging', author='Assignment 7')
    doc.build(story)
    print(f'Report written to {PDF_PATH}')


if __name__ == '__main__':
    build()
