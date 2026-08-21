"""Run the DFA (word-form recognizer) and FST (noun morphology analyzer)
over the Lab 3 examples and the full Brown-corpus noun list."""
import json
from pathlib import Path

from dfa import classify as dfa_classify
from fst import analyze as fst_analyze

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / 'brown_nouns.txt'
OUTPUT_DIR = ROOT / 'output'

DFA_EXAMPLES = [
    ('cat', 'Accepted'),
    ('dog', 'Accepted'),
    ('a', 'Accepted'),
    ('zebra', 'Accepted'),
    ('dog1', 'Not Accepted'),
    ('1dog', 'Not Accepted'),
    ('DogHouse', 'Not Accepted'),
    ('Dog_house', 'Not Accepted'),
    (' cats', 'Not Accepted'),  # starts with a space
]

FST_EXAMPLES = [
    ('foxes', 'fox+N+PL'),
    ('fox', 'fox+N+SG'),
    ('watches', 'watch+N+PL'),
    ('tries', 'try+N+PL'),
    ('bags', 'bag+N+PL'),
    ('foxs', 'Invalid Word'),
]


def run_dfa_examples():
    results = [{'input': w, 'expected': exp, 'actual': dfa_classify(w), 'match': dfa_classify(w) == exp}
               for w, exp in DFA_EXAMPLES]
    return results


def run_fst_examples(lexicon):
    results = [{'input': w, 'expected': exp, 'actual': fst_analyze(w, lexicon), 'match': fst_analyze(w, lexicon) == exp}
               for w, exp in FST_EXAMPLES]
    return results


def main():
    raw_tokens = [w.strip() for w in CORPUS_PATH.read_text(encoding='utf-8').splitlines() if w.strip()]
    unique_words = sorted(set(raw_tokens))
    lexicon = set(w.lower() for w in unique_words if dfa_classify(w) == 'Accepted')

    dfa_example_results = run_dfa_examples()
    fst_example_results = run_fst_examples(lexicon)

    dfa_word_results = {w: dfa_classify(w) for w in unique_words}
    dfa_accepted = [w for w, verdict in dfa_word_results.items() if verdict == 'Accepted']
    dfa_rejected = [w for w, verdict in dfa_word_results.items() if verdict == 'Not Accepted']

    fst_word_results = {w: fst_analyze(w, lexicon) for w in dfa_accepted}
    pl_count = sum(1 for a in fst_word_results.values() if a.endswith('+N+PL'))
    sg_count = sum(1 for a in fst_word_results.values() if a.endswith('+N+SG'))
    invalid_count = sum(1 for a in fst_word_results.values() if a == 'Invalid Word')

    trace_examples = []
    for w, _ in FST_EXAMPLES:
        trace = []
        analysis = fst_analyze(w, lexicon, trace=trace)
        trace_examples.append({'input': w, 'analysis': analysis, 'states_visited': trace})

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / 'dfa_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'example_tests': dfa_example_results,
            'corpus_tokens_checked': len(unique_words),
            'accepted': len(dfa_accepted),
            'not_accepted': len(dfa_rejected),
            'sample_not_accepted': dfa_rejected[:20],
        }, f, indent=2)

    with open(OUTPUT_DIR / 'fst_results.json', 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(fst_word_results.items())), f, indent=2)

    with open(OUTPUT_DIR / 'fst_sample_predictions.json', 'w', encoding='utf-8') as f:
        json.dump({
            'example_tests': fst_example_results,
            'state_traces': trace_examples,
            'summary': {
                'words_analyzed': len(fst_word_results),
                'plural': pl_count,
                'singular': sg_count,
                'invalid': invalid_count,
            },
        }, f, indent=2)

    print(f"Brown noun corpus: {len(raw_tokens)} tokens, {len(unique_words)} unique word types\n")

    print("DFA (Q1) example tests:")
    for r in dfa_example_results:
        mark = 'OK' if r['match'] else 'MISMATCH'
        print(f"  {r['input']!r:<14} -> {r['actual']:<13} [{mark}]")
    print(f"  Corpus check: {len(dfa_accepted)} accepted, {len(dfa_rejected)} not accepted "
          f"(non-lowercase-word tokens, e.g. {dfa_rejected[:5]})\n")

    print("FST (Q2) example tests:")
    for r in fst_example_results:
        mark = 'OK' if r['match'] else 'MISMATCH'
        print(f"  {r['input']!r:<10} -> {r['actual']:<16} [{mark}]")
    print(f"\n  Full corpus (DFA-accepted words only): {len(fst_word_results)} analyzed"
          f" -- {pl_count} plural, {sg_count} singular, {invalid_count} invalid")

    print("\nFull numbers in output/dfa_results.json and output/fst_results.json; "
          "qualitative examples with state traces in output/fst_sample_predictions.json.")


if __name__ == '__main__':
    main()
