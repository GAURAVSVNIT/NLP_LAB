"""Trains the Naive Bayes tagger and the HMM on the Hindi train split, then
evaluates four decoders on the dev and test splits (per-tag F1, accuracy, macro-F1).
Writes output/results.json.
"""
import json
import time
from collections import Counter

from data import OUTPUT_DIR, load_splits
from hmm import HMMTagger
from metrics import evaluate
from naive_bayes import NaiveBayesTagger


def main():
    splits = load_splits()
    train = splits['train']
    train_tags = Counter(t for sent in train for _, t in sent)
    print(f"Train: {len(train)} sentences, {sum(train_tags.values())} tokens, {len(train_tags)} tags")

    print('Fitting Naive Bayes (alpha=1)...')
    nb = NaiveBayesTagger(alpha=1.0).fit(train)
    print('Fitting HMM (alpha=1)...')
    hmm = HMMTagger(alpha=1.0).fit(train)

    decoders = {
        'naive_bayes': nb.predict,
        'hmm_forward_viterbi': hmm.forward_viterbi,
        'hmm_backward_viterbi': hmm.backward_viterbi,
        'hmm_forward_backward_viterbi': hmm.forward_backward_viterbi,
    }

    results = {
        'corpus': {
            'train_sentences': len(train),
            'train_tokens': sum(train_tags.values()),
            'tags': sorted(train_tags),
            'tag_counts_train': dict(train_tags.most_common()),
            'dev_sentences': len(splits['dev']),
            'test_sentences': len(splits['test']),
        },
        'methods': {},
    }

    for name, decode in decoders.items():
        results['methods'][name] = {}
        for split in ('dev', 'test'):
            gold, pred = [], []
            start = time.time()
            for sent in splits[split]:
                words = [w for w, _ in sent]
                gold.extend(t for _, t in sent)
                pred.extend(decode(words))
            metrics = evaluate(gold, pred)
            metrics['seconds'] = round(time.time() - start, 2)
            results['methods'][name][split] = metrics
            print(f"{name:30s} {split:5s} acc={metrics['accuracy']:.4f} "
                  f"macro-F1={metrics['macro_f1']:.4f} ({metrics['seconds']}s)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / 'results.json'
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {out}')
    return results


if __name__ == '__main__':
    main()
