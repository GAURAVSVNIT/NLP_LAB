"""Runs the complete Lab-8 pipeline:
1. Solves the five midsem questions programmatically (output/midsem_solutions.json).
2. Evaluates the noisy-channel spell corrector on spell-errors.txt (output/spell_results.json).
3. Generates the charts in assets/.
4. Compiles the formatted lab report U24AI026_NLP_LAB-8.pdf.
"""
import sys
from pathlib import Path

import midsem
import report
import spell_eval
import visualize

ROOT = Path(__file__).resolve().parent.parent
SPELL_RESULTS = ROOT / 'output' / 'spell_results.json'


def main():
    print('=== NLP Lab 8: Midsem Programs & Noisy-Channel Spell Correction ===', flush=True)
    midsem.main()
    if '--retrain' in sys.argv or not SPELL_RESULTS.exists():
        spell_eval.main()
    else:
        print(f'Loaded existing spell-correction evaluation from {SPELL_RESULTS}')
        print('(Pass --retrain to re-run candidate generation and evaluation, ~12 min)')

    print('\nGenerating charts...', flush=True)
    visualize.main()

    print('\nBuilding PDF report (U24AI026_NLP_LAB-8.pdf)...', flush=True)
    report.build()
    print('\nPipeline execution finished successfully!')


if __name__ == '__main__':
    main()
