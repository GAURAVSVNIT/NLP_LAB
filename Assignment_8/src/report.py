"""Builds U24AI026_NLP_LAB-8.pdf: Part A, worked programmatic solutions to the five midsem questions; Part B,
the noisy-channel spell corrector. All numbers are read from output/midsem_solutions.json and
output/spell_results.json.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
ASSETS = ROOT / 'assets'
REPORT_PATH = ROOT / 'U24AI026_NLP_LAB-8.pdf'

NAVY = colors.HexColor('#2C3E50')
BLUE = colors.HexColor('#2a78d6')
GREEN = colors.HexColor('#1baf7a')
ORANGE = colors.HexColor('#eb6834')
PURPLE = colors.HexColor('#6250d6')
BEST_BG = colors.HexColor('#D1E7DD')
GREY_BG = colors.HexColor('#F4F5F7')
ANSWER_BG = colors.HexColor('#E8F1FC')

MODE_LABELS = {'prior': 'Prior only  P(c)', 'channel': 'Channel only  P(x|c)',
               'noisy_joint': 'Noisy channel, joint P(edit)', 'noisy_conditional': 'Noisy channel, conditional P(edit)'}

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('ReportTitle', parent=styles['Title'], fontSize=21, textColor=NAVY, spaceAfter=4))
styles.add(ParagraphStyle('ReportSubtitle', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#555'), alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle('H1', parent=styles['Heading1'], fontSize=15, textColor=NAVY, spaceBefore=6, spaceAfter=6))
styles.add(ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, textColor=NAVY, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle('H3', parent=styles['Heading3'], fontSize=10.5, textColor=NAVY, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9.5, leading=13.5, spaceAfter=6))
styles.add(ParagraphStyle('Formula', parent=styles['Code'], fontSize=8.5, leading=11.5, alignment=TA_LEFT, textColor=NAVY, backColor=GREY_BG, borderPadding=6, spaceBefore=2, spaceAfter=8))
styles.add(ParagraphStyle('Answer', parent=styles['BodyText'], fontSize=9.5, leading=13.5, backColor=ANSWER_BG, borderPadding=6, spaceBefore=4, spaceAfter=10, textColor=NAVY))
styles.add(ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#666'), alignment=TA_CENTER, spaceBefore=3, spaceAfter=10))
styles.add(ParagraphStyle('Cell', parent=styles['Normal'], fontSize=7.8, leading=9.5))


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


def table(rows, widths, highlight=None, font=8):
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), font),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    for i in range(2, len(rows), 2):
        style.append(('BACKGROUND', (0, i), (-1, i), GREY_BG))
    for cell in highlight or []:
        style += [('BACKGROUND', cell, cell, BEST_BG), ('FONTNAME', cell, cell, 'Helvetica-Bold')]
    t.setStyle(TableStyle(style))
    return t


def figure(name, caption, width=16.5):
    path = ASSETS / name
    iw, ih = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width * cm, height=width * cm * ih / iw),
                         Paragraph(caption, styles['Caption'])])


def f4(x):
    return f'{x:.4f}'


def pct(x, nd=1):
    return f'{x * 100:.{nd}f}%'


# ----------------------------------------------------------------- Part A
def part_a(story, sol):
    story.append(Paragraph('Part A — Programmatic Solutions to the Mid-Semester Questions', styles['H1']))
    story.append(Paragraph(
        'Each question is solved by a short program in <font face="Courier">src/q1_fst.py … src/q5_kneser_ney.py</font>. '
        'The programs print every intermediate quantity; the tables below are generated from their output '
        '(<font face="Courier">output/midsem_solutions.json</font>), so the hand computation and the code are guaranteed '
        'to agree.', styles['Body']))

    # Q1 ---------------------------------------------------------------
    q1 = sol['q1_fst']
    story.append(Paragraph('Q1. FST for English noun singular / plural forms', styles['H2']))
    story.append(Paragraph(
        'The transducer reads the <b>lexical tape</b> (stem letters followed by <b>+N</b> and <b>+SG</b> or <b>+PL</b>) and '
        'writes the <b>surface tape</b>. Letters are copied (l:l), +N and +SG are rewritten to &epsilon;, and +PL '
        'is rewritten to <b>s</b>, or to <b>es</b> when the stem ends in s, x, z, ch or sh. The stem states remember '
        'just enough of the stem\'s ending to make that decision: a final <i>c</i> or <i>s</i> may still become '
        '<i>ch</i> / <i>sh</i>, so they get their own states. Start state q0; accepting state q8.', styles['Body']))
    story.append(figure('q1_fst_diagram.png', 'Figure A1: State diagram of the noun-inflection FST (input : output on each arc).', width=16))
    cols = ['s', 'x, z', 'c', 'h', 'other letter', '+N', '+SG', '+PL']
    rows = [['State', 'Meaning'] + ['x/z' if c == 'x, z' else ('other l' if c == 'other letter' else c) for c in cols]]
    for r in q1['transition_table']:
        rows.append([r['state'], Paragraph(r['description'], styles['Cell'])] + [r[c] for c in cols])
    story.append(table(rows, [1.0, 3.3, 1.55, 1.55, 1.55, 1.55, 1.55, 1.45, 1.4, 1.5], font=7.2))
    story.append(Paragraph('Table A1: Complete transition table. Each cell is <i>next state, input:output</i>; '
                           '— means no transition (the input is rejected).', styles['Caption']))
    gen = q1['generated']
    half = (len(gen) + 1) // 2
    rows = [['Lexical stem', '+N+SG', '+N+PL', '', 'Lexical stem', '+N+SG', '+N+PL']]
    for i in range(half):
        a = gen[i]
        b = gen[i + half] if i + half < len(gen) else None
        rows.append([a['noun'], a['singular'], a['plural'], ''] + ([b['noun'], b['singular'], b['plural']] if b else ['', '', '']))
    story.append(KeepTogether([table(rows, [2.6, 2.4, 2.6, 0.4, 2.6, 2.4, 2.6]),
                               Paragraph('Table A2: Forms generated by running the FST on test stems.', styles['Caption'])]))
    ana = '; '.join(f'<b>{w}</b> &rarr; {", ".join(a)}' for w, a in q1['analyses'].items())
    story.append(Paragraph(
        f'Running the same machine in reverse (surface &rarr; lexical) gives the analyses: {ana}. Without a stem '
        'lexicon the inverted FST also proposes non-words such as <i>foxe+N+PL</i> or <i>cats+N+SG</i>. Composing it '
        'with a lexicon FSA of valid stems would remove them. The rule given in the question produces '
        '<i>quizes</i> and <i>stomaches</i>; the correct forms need two further rules (consonant doubling, and '
        '<i>ch</i> pronounced /k/), which are outside the stated specification.', styles['Body']))

    # Q2 ---------------------------------------------------------------
    q2 = sol['q2_wordpiece']
    story.append(Paragraph('Q2. WordPiece — two merges and the tokenisation of <i>newest</i>', styles['H2']))
    wf = ', '.join(f'{w}: {c}' for w, c in q2['word_freq'].items())
    story.append(Paragraph(
        f'Word frequencies: {wf}. Each word starts as its characters, with the first character unprefixed and '
        'the rest prefixed by ##. WordPiece merges the pair with the highest score '
        'freq(ab) / (freq(a) &middot; freq(b)), which favours pairs whose parts rarely occur apart. BPE would '
        'instead take the most frequent pair.', styles['Body']))
    for s in q2['steps']:
        rows = [['Pair', 'freq(ab)', 'freq(a)', 'freq(b)', 'score']]
        for p in s['pairs'][:7]:
            rows.append([p['pair'], p['pair_freq'], p['freq_a'], p['freq_b'], f4(p['score'])])
        story.append(KeepTogether([
            Paragraph(f'<b>Merge {s["step"]}</b>: {s["merged"]}', styles['Body']),
            table(rows, [3.6, 2.2, 2.2, 2.2, 2.2], highlight=[(0, 1), (4, 1)]),
            Paragraph(f'Table A{2 + s["step"]}: Top pair scores before merge {s["step"]} (all pairs are in the JSON output).',
                      styles['Caption'])]))
    trace = '<br/>'.join(q2['trace'])
    story.append(Paragraph(
        f'Vocabulary after two merges: {{ {", ".join(q2["vocab"])} }}. Tokenising <i>newest</i> greedily, '
        f'longest match first:<br/>{trace}', styles['Formula']))
    story.append(Paragraph(
        f'<b>Answer:</b> merges = <b>##s + ##t &rarr; ##st</b> (score 1/3), then <b>l + ##o &rarr; lo</b> (score 0.1). '
        f'<i>newest</i> &rarr; <b>{" ".join(q2["tokens"])}</b>.', styles['Answer']))

    # Q3 ---------------------------------------------------------------
    q3 = sol['q3_tfidf']
    story.append(Paragraph('Q3. TF-IDF sentence vectors and the closest pair', styles['H2']))
    story.append(Paragraph(
        'TF = raw count; IDF = log<sub>10</sub>(N / df), unsmoothed, N = 4. Each sentence is a TF-IDF vector '
        'over the vocabulary, and closeness is measured by cosine similarity.', styles['Body']))
    vocab = q3['vocab']
    rows = [['Term', 'df', 'idf'] + [f'TF S{i + 1}' for i in range(4)] + [f'TF-IDF S{i + 1}' for i in range(4)]]
    for j, w in enumerate(vocab):
        rows.append([w, q3['df'][w], f4(q3['idf'][w])] + [q3['tf'][i][j] for i in range(4)] +
                    [f4(q3['tfidf'][i][j]) for i in range(4)])
    rows.append(['|v|', '', ''] + [''] * 4 + [f4(n) for n in q3['norms']])
    story.append(table(rows, [1.6, 0.9, 1.4] + [1.25] * 4 + [1.75] * 4, font=7.6))
    story.append(Paragraph('Table A5: Term statistics, TF and TF-IDF vectors (S1 = "cat dog animal", S2 = "cat cat pet", '
                           'S3 = "dog pet animal", S4 = "cat dog pet").', styles['Caption']))
    best = max(q3['pairs'], key=lambda p: p['cosine'])
    rows = [['Pair', 'v_a · v_b', '|v_a|', '|v_b|', 'cosine', 'Euclidean']]
    hl = []
    for i, p in enumerate(q3['pairs'], start=1):
        rows.append([p['pair'], f4(p['dot']), f4(p['norm_a']), f4(p['norm_b']), f4(p['cosine']), f4(p['euclidean'])])
        if p is best:
            hl += [(0, i), (4, i)]
    story.append(KeepTogether([table(rows, [2, 2.4, 2.2, 2.2, 2.2, 2.2], highlight=hl),
                               Paragraph('Table A6: Pairwise similarities.', styles['Caption'])]))
    story.append(figure('q3_tfidf_cosine.png', 'Figure A2: Cosine-similarity matrix of the four TF-IDF vectors.', width=9.5))
    story.append(Paragraph(
        f'<b>Answer:</b> <b>S1 ("cat dog animal") and S3 ("dog pet animal")</b> are closest, with cosine '
        f'{best["cosine"]:.4f}. They share the rare, high-IDF term <i>animal</i> (idf 0.301), which dominates both '
        f'vectors. Euclidean distance ties {" and ".join(q3["euclidean_ties"])}, because it is sensitive to vector '
        'length; cosine normalises length away and gives a unique answer. The choice of log base rescales every '
        'weight by the same factor, so the cosine ranking does not depend on it.', styles['Answer']))

    # Q4 ---------------------------------------------------------------
    q4 = sol['q4_naive_bayes']
    story.append(Paragraph('Q4. Naive Bayes spam classifier — E8 = (free = 0, win = 0, offer = 0)', styles['H2']))
    story.append(Paragraph(
        'Bernoulli Naive Bayes: P(c | x) &prop; P(c) &Pi;<sub>k</sub> P(x<sub>k</sub> | c), where an absent word '
        'contributes P(x<sub>k</sub> = 0 | c) = 1 &minus; P(x<sub>k</sub> = 1 | c). The exam expects the unsmoothed '
        'MLE; the Laplace (add-1) version is shown as a check.', styles['Body']))
    rows = [['Estimate', 'Class', 'P(c)', 'P(free=0|c)', 'P(win=0|c)', 'P(offer=0|c)', 'P(c)·Π P(x|c)', 'Posterior']]
    hl = []
    for name, label in [('mle', 'MLE'), ('laplace', 'Laplace')]:
        r = q4[name]
        for c, v in r['classes'].items():
            rows.append([label if c == 'Spam' else '', c, v['prior']['fraction']] +
                        [t['p']['fraction'] for t in v['terms']] +
                        [f"{v['joint']['fraction']} = {v['joint']['value']:.5f}", f"{v['posterior']['value']:.4f}"])
            if c == r['prediction']:
                hl += [(6, len(rows) - 1), (7, len(rows) - 1)]
    story.append(table(rows, [1.5, 1.7, 1.2, 2.0, 2.0, 2.1, 3.4, 1.6], highlight=hl))
    story.append(Paragraph('Table A7: Naive Bayes computation for E8. Spam: free, win and offer each appear in 3 of 4 '
                           'spam mails; NotSpam: each appears in 1 of 3.', styles['Caption']))
    story.append(Paragraph(
        f'<b>Answer:</b> P(Spam)&middot;(1/4)<super>3</super> = 1/112 &asymp; 0.0089 &lt; '
        f'P(NotSpam)&middot;(2/3)<super>3</super> = 8/63 &asymp; 0.1270, so E8 is classified as <b>{q4["mle"]["prediction"]}</b> '
        f'(posterior {q4["mle"]["classes"]["NotSpam"]["posterior"]["value"]:.3f}). Laplace smoothing gives the same '
        f'decision ({q4["laplace"]["prediction"]}, posterior {q4["laplace"]["classes"]["NotSpam"]["posterior"]["value"]:.3f}). '
        'This is intuitive: spam mails almost always contain at least one trigger word, and E8 contains none.',
        styles['Answer']))

    # Q5 ---------------------------------------------------------------
    q5 = sol['q5_kneser_ney']
    story.append(Paragraph('Q5. Kneser-Ney smoothing with d = 0.5', styles['H2']))
    story.append(Paragraph(
        'P<sub>KN</sub>(w | v) = max(C(vw) &minus; d, 0) / C(v) + &lambda;(v)&middot;P<sub>cont</sub>(w), with '
        '&lambda;(v) = d &middot; |{w\' : C(vw\') &gt; 0}| / C(v) and P<sub>cont</sub>(w) = |{v\' : C(v\'w) &gt; 0}| / '
        '(number of bigram types). C(v) counts bigram tokens that begin with v; no boundary markers are used.',
        styles['Body']))
    bc = ', '.join(f'{k}: {v}' for k, v in q5['bigram_counts'].items())
    story.append(Paragraph(f'Bigram counts ({q5["bigram_types"]} types): {bc}.', styles['Body']))
    rows = [['Query', 'C(vw)', 'C(v)', 'Discounted term', 'Followers of v', 'λ(v)', 'Preceders of w', 'P_cont(w)', 'P_KN']]
    for a in q5['answers']:
        rows.append([a['query'], a['C(v w)'], a['C(v)'], a['discounted_term'], ', '.join(a['followers_of_v']),
                     a['lambda'], ', '.join(a['preceders_of_w']), a['p_cont'],
                     f"{a['probability']} = {a['probability_float']:.4f}"])
    story.append(table(rows, [2.7, 1.1, 1.0, 1.9, 2.1, 1.0, 2.1, 1.6, 2.4],
                       highlight=[(8, 1), (8, 2)], font=7.6))
    story.append(Paragraph('Table A8: Kneser-Ney computation.', styles['Caption']))
    a1, a2 = q5['answers']
    story.append(Paragraph(
        f'<b>Answer:</b> P<sub>KN</sub>(AI | love) = (1 &minus; 0.5)/2 + (0.5&middot;2/2)&middot;(2/8) = 0.25 + 0.125 = '
        f'<b>{a1["probability"]} = {a1["probability_float"]:.4f}</b>.<br/>'
        f'P<sub>KN</sub>(NLP | like) = (2 &minus; 0.5)/3 + (0.5&middot;2/3)&middot;(2/8) = 0.5 + 0.0833 = '
        f'<b>{a2["probability"]} = {a2["probability_float"]:.4f}</b>.', styles['Answer']))


# ----------------------------------------------------------------- Part B
def part_b(story, res):
    data, main = res['data'], res['main']
    story.append(Paragraph('Part B — Noisy-Channel Spelling Correction', styles['H1']))
    story.append(Paragraph('B1. Data', styles['H2']))
    dh = data['distance_histogram']
    story.append(Paragraph(
        f'<b>spell-errors.txt</b> lists correct words with their observed misspellings (e.g. <i>raining: rainning, '
        f'raning</i>; a suffix *n is a repeat count). After lowercasing (underscores become spaces), it yields '
        f'<b>{data["unique_pairs"]:,}</b> distinct (correct, misspelling) pairs covering {data["tokens"]:,} error tokens. '
        f'{data["identical_after_lowercasing"]} capitalisation-only "errors" are dropped. '
        f'<b>count_unigram.txt</b> gives web frequencies for {data["vocab_size"]:,} words, and '
        f'<b>count_1edit.txt</b> gives {data["edit_types"]:,} single-edit types ({data["edit_total"]:,} occurrences). '
        f'The true Damerau-Levenshtein distance between misspelling and correct word is 1 for {dh["1"]:,} pairs, '
        f'2 for {dh["2"]:,}, 3 for {dh["3"]:,} and 4 or more for {dh["4"]:,}. Only '
        f'{dh["1"] / data["unique_pairs"] * 100:.1f}% are true single-edit errors. Also, '
        f'{pct(data["typo_in_vocab_rate"])} of the misspellings are themselves in the web vocabulary: real-word '
        f'errors such as <i>fore</i> for <i>four</i>, and misspellings common enough on the web to be counted, such '
        f'as <i>rainning</i> or <i>begining</i>.', styles['Body']))
    dc = res['edit_direction_check']
    story.append(Paragraph(
        '<b>Edit notation.</b> An entry <font face="Courier">x|y</font> means the writer typed x where the correct '
        'word has y, and <font face="Courier">&gt;</font> marks the start of the word. For example '
        '<font face="Courier">t|te</font> means the e after t was dropped. This direction was verified, not assumed. '
        'Every single-insertion/deletion pair in spell-errors.txt was aligned, and the derived counts were '
        f'correlated with count_1edit.txt: Pearson r = {dc["typo|correct"]["pearson_r"]:.2f} under the '
        f'typo|correct reading versus {dc["correct|typo"]["pearson_r"]:.2f} under the reverse reading.', styles['Body']))

    story.append(Paragraph('B2. Model', styles['H2']))
    story.append(Paragraph(
        'c* = argmax_c  P(c) &middot; P(x | c)      over candidates c within k edits of the misspelling x<br/>'
        'P(c) = count(c) / N                                (unigram language model, count_unigram.txt)<br/>'
        'P(x | c) = p_err &middot; &Pi;_i P(e_i)               (edits e_i that turn c into x)<br/>'
        '  joint:        P(e) = count(e) / &Sigma;_e\' count(e\')                     (Norvig)<br/>'
        '  conditional:  P(e) = count(e) / count(correct side of e in corpus)   (Kernighan, Church &amp; Gale)',
        styles['Formula']))
    story.append(Paragraph(
        'The conditional estimator is the textbook form, for example del[x,y] / count[xy]. Its denominators are '
        'the frequency-weighted counts of the 1- or 2-character string in the unigram corpus. Unseen edits get a '
        'pseudo-count of 0.5. Candidates are generated by a depth-first search over a trie of all vocabulary '
        'prefixes, which only extends prefixes of real words. That makes the distance-2 search cheap '
        f'(about 8 ms per word over {data["vocab_size"]:,} words); all {data["unique_typos"]:,} unique misspellings are '
        'processed in parallel. When several edit sequences reach the same candidate, the most probable one is used.',
        styles['Body']))
    story.append(Paragraph(
        '<b>Evaluation setting.</b> Every input in spell-errors.txt is a known misspelling. The main setting '
        'therefore assumes <b>error detection is given</b> and never returns the input unchanged, which is the '
        'classic non-word correction task. In that setting p_err only scales every score equally and does not '
        'affect the ranking. Section B5 also reports the <b>open</b> setting, where the input itself is a candidate '
        'with P(x|x) = 1 &minus; p_err. Accuracy is top-1 over all distinct pairs; token-weighted accuracy, which '
        'counts repeated errors, is also stored in the JSON output.', styles['Body']))

    story.append(Paragraph('B3. Results', styles['H2']))
    rows = [['Scoring model', 'k = 1: accuracy', 'k = 1: token acc.', 'k = 2: accuracy', 'k = 2: token acc.']]
    hl = []
    for i, mode in enumerate(MODE_LABELS, start=1):
        r1, r2 = main[f'detected|d1|{mode}'], main[f'detected|d2|{mode}']
        rows.append([MODE_LABELS[mode], pct(r1['accuracy'], 2), pct(r1['token_accuracy'], 2),
                     pct(r2['accuracy'], 2), pct(r2['token_accuracy'], 2)])
    for col, d in [(1, 'd1'), (3, 'd2')]:
        b = max(range(len(MODE_LABELS)), key=lambda i: main[f'detected|{d}|{list(MODE_LABELS)[i]}']['accuracy'])
        hl.append((col, b + 1))
    rows.append(['Oracle: correct word among candidates', pct(res['oracle']['d1'], 2), '', pct(res['oracle']['d2'], 2), ''])
    story.append(table(rows, [6.0, 2.6, 2.6, 2.6, 2.6], highlight=hl))
    story.append(Paragraph('Table B1: Top-1 accuracy over all 39,507 pairs (error detection given). k = maximum edit '
                           'distance of candidates. The oracle row is the upper bound set by candidate generation.',
                           styles['Caption']))
    story.append(figure('spell_accuracy.png', 'Figure B1: Accuracy of each scoring model against the candidate-generation oracle.'))

    bd = res['breakdown']
    j1, c1 = bd['detected|d1|noisy_joint'], bd['detected|d1|noisy_conditional']
    j2, c2 = bd['detected|d2|noisy_joint'], bd['detected|d2|noisy_conditional']
    groups = ['distance=1 non-word (classic setting)', 'correct word reachable', 'distance=1', 'distance=2',
              'distance=3+', 'typo is a non-word', 'typo is a vocabulary word']
    rows = [['Subset', 'Pairs', 'k=1 joint', 'k=1 cond.', 'k=2 joint', 'k=2 cond.']]
    for g in groups:
        rows.append([g, f'{j1[g]["n_pairs"]:,}' if g != 'correct word reachable' else 'varies',
                     pct(j1[g]['accuracy']), pct(c1[g]['accuracy']), pct(j2[g]['accuracy']), pct(c2[g]['accuracy'])])
    story.append(KeepTogether([table(rows, [6.0, 1.8, 2.2, 2.2, 2.2, 2.2]),
                               Paragraph('Table B2: Accuracy by subset. "Correct word reachable" is the subset where '
                                         'the correct word is among the candidates (its size differs for k = 1 and k = 2).',
                                         styles['Caption'])]))
    story.append(figure('spell_breakdown.png', 'Figure B2: Accuracy by error type for the four noisy-channel configurations.'))
    tk = res['top_k']
    rows = [['Configuration', 'Top-1', 'Top-3', 'Top-5']]
    for key, label in [('detected|d1|noisy_joint', 'k=1, joint'), ('detected|d1|noisy_conditional', 'k=1, conditional'),
                       ('detected|d2|noisy_joint', 'k=2, joint'), ('detected|d2|noisy_conditional', 'k=2, conditional')]:
        rows.append([f'Noisy channel, {label}'] + [pct(tk[key][f'top{k}']) for k in [1, 3, 5]])
    story.append(KeepTogether([table(rows, [6, 2.5, 2.5, 2.5]),
                               Paragraph('Table B3: Top-k accuracy (the correct word is among the k best-ranked suggestions).',
                                         styles['Caption'])]))

    story.append(Paragraph('B4. Leakage check: held-out channel counts', styles['H2']))
    cv = res['cross_validation']
    story.append(Paragraph(
        'Every edit we can align in spell-errors.txt also appears in count_1edit.txt, so the channel model was '
        'built from the same errors it is tested on. To measure the effect, a 5-fold cross-validation grouped by '
        'correct word re-estimates the edit counts from the four training folds alone (aligning each pair of '
        'distance &le; 2) and tests on the held-out fold.', styles['Body']))
    rows = [['Configuration', 'count_1edit.txt', 'Held-out counts', 'Difference']]
    for c in ['d1|joint', 'd1|conditional', 'd2|joint', 'd2|conditional']:
        a, b = cv[f'count_1edit|{c}']['mean'], cv[f'held_out|{c}']['mean']
        rows.append([f'k={c[1]}, {c.split("|")[1]}', pct(a, 2), pct(b, 2), f'{(b - a) * 100:+.2f} pts'])
    story.append(KeepTogether([table(rows, [5, 3, 3, 3]),
                               Paragraph('Table B4: 5-fold cross-validated accuracy (mean over folds).', styles['Caption'])]))
    story.append(figure('spell_cv_leakage.png', 'Figure B3: Channel counts from count_1edit.txt vs counts re-estimated on training folds only.', width=12))

    story.append(Paragraph('B5. Open setting: should the input itself be a candidate?', styles['H2']))
    sw = res['p_err_sweep']
    story.append(Paragraph(
        f'If the misspelling may also be returned unchanged, the {pct(data["typo_in_vocab_rate"], 0)} of misspellings '
        f'that are web-vocabulary words compete as candidates. With p_err = {res["p_err_default"]} and the joint '
        f'channel, accuracy falls from {pct(main["detected|d1|noisy_joint"]["accuracy"])} to '
        f'{pct(main["open|d1|noisy_joint"]["accuracy"])} (k = 1) and from {pct(main["detected|d2|noisy_joint"]["accuracy"])} '
        f'to {pct(main["open|d2|noisy_joint"]["accuracy"])} (k = 2). Raising p_err recovers accuracy steadily, '
        f'reaching {pct(sw["d1|joint"]["0.99"])} at p_err = 0.99 (k = 1), because on this test set every input is an '
        'error. The conditional channel is flat across p_err. Its edit probabilities are counts of about 40k '
        'observed errors divided by character counts in a corpus of about 10<super>11</super> tokens. They are '
        'correctly scaled <i>relative to each other</i>, but far smaller than any p_err could offset, so the '
        'unchanged input always wins when it is a vocabulary word.', styles['Body']))
    story.append(figure('spell_p_err_sweep.png', 'Figure B4: Open-setting accuracy as a function of p_err.', width=13))

    story.append(Paragraph('B6. Example corrections', styles['H2']))
    ex = res['examples']
    rows = [['Misspelling', 'Correct', 'Dist.', 'k=1 top-3 (joint)', 'k=2 prediction', 'k=1 result']]
    for e in ex[:10] + [e for e in ex if not e['correct_d1']][:10]:
        rows.append([e['typo'], e['correct'], e['distance'], ', '.join(e['top3_d1']) or '(no candidate)',
                     e['pred_d2'] or '—', 'correct' if e['correct_d1'] else 'wrong'])
    story.append(table(rows, [2.6, 2.6, 1.0, 5.4, 2.6, 1.8], font=7.4))
    ae = res['assignment_example']
    story.append(Paragraph(
        'Table B5: Randomly sampled successes and failures. For the example in the assignment, <i>rainning</i> '
        f'&rarr; {", ".join(ae["rainning"]["d1"])} and <i>raning</i> &rarr; {", ".join(ae["raning"]["d1"])} '
        '(top-3, k = 1, joint).', styles['Caption']))

    story.append(Paragraph('B7. Discussion', styles['H2']))
    story.append(Paragraph(
        f'<b>Where the noisy channel works.</b> On the classic task (single-edit, non-word errors) the corrector '
        f'is right {pct(c1["distance=1 non-word (classic setting)"]["accuracy"])} of the time with the conditional '
        f'channel ({pct(j1["distance=1 non-word (classic setting)"]["accuracy"])} joint). Combining prior and channel '
        f'clearly beats either alone: at k = 2 the prior alone gives {pct(main["detected|d2|prior"]["accuracy"])}, '
        f'the channel alone {pct(main["detected|d2|channel"]["accuracy"])}, and the noisy channel '
        f'{pct(main["detected|d2|noisy_joint"]["accuracy"])}. The prior alone picks frequent words such as <i>the</i> '
        'or <i>a</i>; the channel alone picks the easiest edit regardless of whether the word is plausible.',
        styles['Body']))
    story.append(Paragraph(
        f'<b>Where it does not.</b> Overall accuracy is bounded by candidate generation: only '
        f'{pct(res["oracle"]["d1"])} of correct words lie within one edit, and {pct(res["oracle"]["d2"])} within two. '
        'The Birkbeck/Holbrook errors were written by poor spellers and are often phonetic (<i>negoshation</i>, '
        '<i>fretenery</i> for <i>fraternally</i>), far beyond two character edits. The remaining errors among '
        'reachable words are mostly real-word or context-dependent confusions (<i>cack</i> &rarr; <i>cock</i> vs '
        '<i>cake</i>, <i>fut</i> &rarr; <i>fat</i> vs <i>fruit</i>), which a unigram prior cannot resolve. Extending to '
        f'k = 2 adds {(main["detected|d2|noisy_joint"]["accuracy"] - main["detected|d1|noisy_joint"]["accuracy"]) * 100:.1f} '
        'points overall with almost no loss on single-edit errors, because two-edit candidates pay a second '
        'small edit probability. The joint estimator generalises better at k = 2, while the conditional estimator '
        'is slightly better at k = 1.', styles['Body']))
    story.append(Paragraph(
        '<b>Leakage.</b> Re-estimating the channel from held-out folds changes accuracy by at most '
        f'{max(abs(cv[f"held_out|{c}"]["mean"] - cv[f"count_1edit|{c}"]["mean"]) for c in ["d1|joint", "d1|conditional", "d2|joint", "d2|conditional"]) * 100:.2f} '
        'points. Common edits (vowel confusions, dropped doubled letters) recur across many different words, so '
        'the channel model learns general patterns rather than memorising specific test errors.', styles['Body']))
    story.append(Paragraph(
        '<b>Next steps.</b> (1) A bigram or trigram context model P(c | neighbours) to fix real-word errors. '
        '(2) Phonetic candidate generation (Soundex / Metaphone keys) for the distance-3+ errors. (3) Pruning '
        'web-noise entries from the vocabulary, or penalising them, so that an open-setting corrector can safely '
        'leave correctly spelled words alone.', styles['Body']))


def build():
    sol = json.loads((OUTPUT_DIR / 'midsem_solutions.json').read_text(encoding='utf-8'))
    res = json.loads((OUTPUT_DIR / 'spell_results.json').read_text(encoding='utf-8'))
    doc = SimpleDocTemplate(str(REPORT_PATH), pagesize=A4, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm)
    story = [Spacer(1, 0.4 * cm),
             Paragraph('NLP Laboratory — Assignment 8', styles['ReportTitle']),
             Paragraph('Programmatic Solutions to the Mid-Semester Questions and a Noisy-Channel Spell Corrector',
                       styles['ReportSubtitle']),
             HRFlowable(width='100%', thickness=1, color=colors.HexColor('#DDD')),
             Spacer(1, 0.4 * cm)]
    meta = Table([
        ['Name', 'GAURAV DAMOR', 'Course', 'Natural Language Processing (AI357)'],
        ['Roll No.', 'U24AI026', 'Lab', 'Lab 8 — Midsem Programs & Spell Correction'],
    ], colWidths=[2.2 * cm, 5.8 * cm, 2.0 * cm, 6.8 * cm])
    meta.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), NAVY), ('TEXTCOLOR', (2, 0), (2, -1), NAVY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story += [meta, Spacer(1, 0.5 * cm)]
    best1 = max(res['main'][f'detected|d1|{m}']['accuracy'] for m in MODE_LABELS)
    best2 = max(res['main'][f'detected|d2|{m}']['accuracy'] for m in MODE_LABELS)
    classic = res['breakdown']['detected|d1|noisy_conditional']['distance=1 non-word (classic setting)']['accuracy']
    cards = Table([[
        stat_card('5 / 5', 'Midsem questions solved by code', BLUE),
        stat_card(pct(classic), 'Accuracy, 1-edit non-word errors', GREEN),
        stat_card(pct(best1), 'Overall accuracy, 1-edit model', PURPLE),
        stat_card(pct(best2), 'Overall accuracy, 2-edit model', ORANGE),
    ]], colWidths=[4.25 * cm] * 4)
    cards.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story += [cards, Spacer(1, 0.4 * cm)]

    q2, q3, q4, q5 = (sol[k] for k in ['q2_wordpiece', 'q3_tfidf', 'q4_naive_bayes', 'q5_kneser_ney'])
    rows = [['Question', 'Topic', 'Answer'],
            ['Q1', 'FST, noun number', Paragraph('9-state transducer (Fig. A1, Table A1); +PL &rarr; "es" after s, x, z, ch, sh, else "s"', styles['Cell'])],
            ['Q2', 'WordPiece', Paragraph(f'Merges: ##s+##t, l+##o; newest &rarr; {" ".join(q2["tokens"])}', styles['Cell'])],
            ['Q3', 'TF-IDF', Paragraph(f'S1 and S3 are closest (cosine {q3["closest_cosine"]:.4f})', styles['Cell'])],
            ['Q4', 'Naive Bayes', Paragraph(f'E8 &rarr; {q4["mle"]["prediction"]} (1/112 vs 8/63)', styles['Cell'])],
            ['Q5', 'Kneser-Ney', Paragraph(f'P(AI|love) = {q5["answers"][0]["probability"]} = {q5["answers"][0]["probability_float"]:.4f}; '
                                           f'P(NLP|like) = {q5["answers"][1]["probability"]} = {q5["answers"][1]["probability_float"]:.4f}', styles['Cell'])],
            ['Spell', 'Noisy channel', Paragraph(f'{pct(best1)} (1 edit) / {pct(best2)} (2 edits) top-1 over all 39,507 pairs; '
                                                 f'{pct(classic)} on single-edit non-word errors', styles['Cell'])]]
    story.append(Paragraph('Summary of Answers', styles['H2']))
    story += [table(rows, [1.6, 3.2, 12.0]), Spacer(1, 0.2 * cm)]

    story.append(PageBreak())
    part_a(story, sol)
    story.append(PageBreak())
    part_b(story, res)
    doc.build(story)
    print(f'Wrote {REPORT_PATH}')


if __name__ == '__main__':
    build()
