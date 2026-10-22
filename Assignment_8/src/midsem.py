"""Runs the five midsem solutions and writes output/midsem_solutions.json."""
import json
from pathlib import Path

import q1_fst
import q2_wordpiece
import q3_tfidf
import q4_naive_bayes
import q5_kneser_ney

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    solutions = {
        'q1_fst': q1_fst.solve(),
        'q2_wordpiece': q2_wordpiece.solve(),
        'q3_tfidf': q3_tfidf.solve(),
        'q4_naive_bayes': q4_naive_bayes.solve(),
        'q5_kneser_ney': q5_kneser_ney.solve(),
    }
    path = OUTPUT_DIR / 'midsem_solutions.json'
    path.write_text(json.dumps(solutions, indent=2, default=str), encoding='utf-8')
    q2, q3, q4, q5 = (solutions[k] for k in ['q2_wordpiece', 'q3_tfidf', 'q4_naive_bayes', 'q5_kneser_ney'])
    print(f"Q2 WordPiece: merges = {[s['merged'] for s in q2['steps']]}, newest -> {' '.join(q2['tokens'])}")
    print(f"Q3 TF-IDF: closest pair = {q3['closest']} (cosine {q3['closest_cosine']:.4f})")
    print(f"Q4 Naive Bayes: E8 -> {q4['mle']['prediction']} (Laplace: {q4['laplace']['prediction']})")
    for a in q5['answers']:
        print(f"Q5 {a['query']} = {a['probability']} = {a['probability_float']:.4f}")
    print(f'Wrote {path}')
    return solutions


if __name__ == '__main__':
    main()
