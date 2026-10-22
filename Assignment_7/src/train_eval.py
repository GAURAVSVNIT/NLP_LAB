"""Trains the Naive Bayes and HMM taggers on the training split and evaluates every algorithm on the
dev and test splits. Writes output/results.json and output/confusion_<split>.json.
"""
import json
import time
from collections import Counter
from pathlib import Path

from data import load_all
from hmm import HMMTagger
from metrics import confusion_matrix, evaluate
from naive_bayes import NaiveBayesTagger

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

ALGORITHMS = ['naive_bayes', 'forward_viterbi', 'backward_viterbi', 'forward_backward_viterbi']


def run_tagger(fn, sentences):
    gold, pred = [], []
    start = time.perf_counter()
    for sent in sentences:
        words = [w for w, _ in sent]
        pred.extend(fn(words, sent))
        gold.extend(t for _, t in sent)
    return gold, pred, time.perf_counter() - start


NB_ALPHAS = [1.0, 0.3, 0.1, 0.03, 0.01]
HMM_KS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def tune(train, dev):
    """Chooses the smoothing hyper-parameters on the dev split only (test is never consulted)."""
    tuning = {'nb_alpha': {}, 'hmm_k': {}}
    for a in NB_ALPHAS:
        nb = NaiveBayesTagger(alpha=a).fit(train)
        gold, pred, _ = run_tagger(lambda w, s: nb.tag(w), dev)
        tuning['nb_alpha'][str(a)] = accuracy(gold, pred)
    for k in HMM_KS:
        hmm = HMMTagger(k_transition=k).fit(train)
        row, outputs = {}, {}
        for name in ['forward_viterbi', 'backward_viterbi', 'forward_backward_viterbi']:
            gold, outputs[name], _ = run_tagger(lambda w, s: getattr(hmm, name)(w), dev)
            row[name] = accuracy(gold, outputs[name])
        row['forward_vs_backward_token_diffs'] = sum(
            a != b for a, b in zip(outputs['forward_viterbi'], outputs['backward_viterbi']))
        tuning['hmm_k'][str(k)] = row
    best_alpha = max(NB_ALPHAS, key=lambda a: tuning['nb_alpha'][str(a)])
    best_k = max(HMM_KS, key=lambda k: sum(v for n, v in tuning['hmm_k'][str(k)].items() if n.endswith('viterbi')))
    return best_alpha, best_k, tuning


def main():
    data = load_all()
    train = data['train']

    print('Tuning smoothing hyper-parameters on dev...')
    best_alpha, best_k, tuning = tune(train, data['dev'])
    print(f'  Naive Bayes alpha = {best_alpha}, HMM transition add-k = {best_k}')

    print('Training Naive Bayes...')
    nb = NaiveBayesTagger(alpha=best_alpha).fit(train)
    print('Training HMM...')
    hmm = HMMTagger(k_transition=best_k).fit(train)
    tags = hmm.tags

    # Numerical check of the forward/backward joint-probability identity on gold dev sequences.
    max_gap = max(
        abs(hmm.joint_logprob([w for w, _ in s], [t for _, t in s], 'forward')
            - hmm.joint_logprob([w for w, _ in s], [t for _, t in s], 'backward'))
        for s in data['dev'])

    taggers = {
        'naive_bayes': lambda words, sent: nb.tag(words),
        'forward_viterbi': lambda words, sent: hmm.forward_viterbi(words),
        'backward_viterbi': lambda words, sent: hmm.backward_viterbi(words),
        'forward_backward_viterbi': lambda words, sent: hmm.forward_backward_viterbi(words),
    }
    analysis_taggers = {
        'naive_bayes_gold_prev_tag': lambda words, sent: nb.tag(words, gold_tags=[t for _, t in sent]),
        'max_marginal_check': lambda words, sent: hmm.max_marginal_check(words),
    }

    train_tags = Counter(t for s in train for _, t in s)
    results = {
        'corpus': {
            split: {'sentences': len(s), 'tokens': sum(len(x) for x in s)} for split, s in data.items()
        },
        'tags': tags,
        'train_tag_counts': dict(train_tags.most_common()),
        'vocab_size': len(hmm.vocab),
        'hyperparameters': {'nb_laplace_alpha': nb.alpha, 'hmm_transition_add_k': hmm.k},
        'tuning_dev': tuning,
        'forward_backward_joint_max_abs_logprob_gap': max_gap,
        'splits': {},
        'analysis': {},
    }

    for split in ['dev', 'test']:
        sents = data[split]
        known = [w in hmm.vocab for s in sents for w, _ in s]
        results['corpus'][split]['oov_rate'] = 1 - sum(known) / len(known)
        split_res, preds = {}, {}
        for name, fn in taggers.items():
            gold, pred, secs = run_tagger(fn, sents)
            preds[name] = pred
            split_res[name] = evaluate(gold, pred, tags, known)
            split_res[name]['seconds'] = secs
            print(f'  [{split}] {name:26s} acc={split_res[name]["accuracy"]:.4f} '
                  f'macroF1={split_res[name]["macro_f1"]:.4f} ({secs:.1f}s)')
        results['splits'][split] = split_res

        analysis = {}
        for name, fn in analysis_taggers.items():
            _, pred, _ = run_tagger(fn, sents)
            analysis[name] = evaluate(gold, pred, tags, known)
            if name == 'max_marginal_check':
                agree = sum(a == b for a, b in zip(pred, preds['forward_viterbi']))
                analysis[name]['agreement_with_forward_viterbi'] = agree / len(pred)
        names = list(taggers)
        analysis['pairwise_agreement'] = {
            f'{a}|{b}': sum(x == y for x, y in zip(preds[a], preds[b])) / len(gold)
            for i, a in enumerate(names) for b in names[i + 1:]
        }
        analysis['sentence_level_disagreements'] = {
            'forward_vs_backward': sum(
                _split_by_sentence(preds['forward_viterbi'], preds['backward_viterbi'], sents)),
        }
        results['analysis'][split] = analysis

        best = max(ALGORITHMS, key=lambda a: split_res[a]['accuracy'])
        (OUTPUT_DIR / f'confusion_{split}.json').write_text(json.dumps({
            'algorithm': best,
            'tags': tags,
            'matrix': confusion_matrix(gold, preds[best], tags),
            'naive_bayes_matrix': confusion_matrix(gold, preds['naive_bayes'], tags),
        }), encoding='utf-8')

    (OUTPUT_DIR / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_DIR / "results.json"}')


def _split_by_sentence(a, b, sents):
    """Yields True for each sentence on which two flat prediction lists differ."""
    pos = 0
    for s in sents:
        n = len(s)
        yield a[pos:pos + n] != b[pos:pos + n]
        pos += n


if __name__ == '__main__':
    main()
