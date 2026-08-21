"""
Tokenize every paragraph of data/raw_corpus.txt into sentences and words,
and save the result in two forms:
  - output/tokenized.jsonl      structured: {"paragraph_id", "sentences": [[tok, ...], ...]}
  - output/tokenized_readable.txt   human-readable: one sentence per line
    (space-separated tokens), blank line between paragraphs.
"""
import json
import sys
import time

sys.path.insert(0, "src")
from tokenizer import tokenize_paragraph

RAW_PATH = r"D:\CODING\NLP\Assignment_1\data\raw_corpus.txt"
JSONL_PATH = r"D:\CODING\NLP\Assignment_1\output\tokenized.jsonl"
READABLE_PATH = r"D:\CODING\NLP\Assignment_1\output\tokenized_readable.txt"


def main():
    start = time.time()
    n_paragraphs = 0

    with open(RAW_PATH, "r", encoding="utf-8") as fin, \
         open(JSONL_PATH, "w", encoding="utf-8", newline="\n") as fjsonl, \
         open(READABLE_PATH, "w", encoding="utf-8", newline="\n") as fread:

        for pid, line in enumerate(fin):
            para = line.strip()
            if not para:
                continue
            sentences = tokenize_paragraph(para)
            if not sentences:
                continue

            fjsonl.write(json.dumps({"paragraph_id": pid, "sentences": sentences}, ensure_ascii=False) + "\n")

            for sent in sentences:
                fread.write(" ".join(sent) + "\n")
            fread.write("\n")

            n_paragraphs += 1
            if n_paragraphs % 20000 == 0:
                elapsed = time.time() - start
                print(f"processed {n_paragraphs} paragraphs in {elapsed:.1f}s", flush=True)

    elapsed = time.time() - start
    print(f"DONE paragraphs={n_paragraphs} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
