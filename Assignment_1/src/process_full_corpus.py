"""
Full-dataset run: streams every row of every parquet shard in
D:\\CODING\\NLP\\Dataset\\indiccorp_v2\\partial-guj_Gujr\\*.parquet directly
into the tokenizer (no intermediate raw-text file, to save disk space),
writes the tokenized output as gzip-compressed JSONL, and accumulates
corpus statistics in the same pass (re-reading ~15M JSON lines back just
to recompute the same numbers would take significantly longer than folding
the counts in while we already have the tokens in hand).

Output (kept separate from the partial-sample run in output/):
  output_full/tokenized.jsonl.gz   one JSON object per paragraph:
                                    {"paragraph_id", "sentences": [[tok,...], ...]}
  output_full/corpus_statistics.txt / .json
"""
import glob
import gzip
import json
import sys
import time

sys.path.insert(0, "src")
from tokenizer import tokenize_paragraph
import pyarrow.parquet as pq

PARQUET_GLOB = r"D:\CODING\NLP\Dataset\indiccorp_v2\partial-guj_Gujr\*.parquet"
JSONL_GZ_PATH = r"D:\CODING\NLP\Assignment_1\output_full\tokenized.jsonl.gz"
STATS_TXT_PATH = r"D:\CODING\NLP\Assignment_1\output_full\corpus_statistics.txt"
STATS_JSON_PATH = r"D:\CODING\NLP\Assignment_1\output_full\corpus_statistics.json"

PROGRESS_EVERY = 500_000


def main():
    files = sorted(glob.glob(PARQUET_GLOB))
    print(f"found {len(files)} parquet shards")

    total_sentences = 0
    total_words = 0
    total_chars = 0
    unique_tokens = set()
    n_paragraphs = 0
    pid = 0

    start = time.time()

    with gzip.open(JSONL_GZ_PATH, "wt", encoding="utf-8", newline="\n", compresslevel=6) as fout:
        for fpath in files:
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
                for t in batch.column("text").to_pylist():
                    pid += 1
                    if t is None:
                        continue
                    para = t.strip().replace("\n", " ").replace("\r", " ")
                    if not para:
                        continue

                    sentences = tokenize_paragraph(para)
                    if not sentences:
                        continue

                    fout.write(json.dumps({"paragraph_id": pid, "sentences": sentences}, ensure_ascii=False) + "\n")

                    for sent in sentences:
                        total_sentences += 1
                        total_words += len(sent)
                        for tok in sent:
                            total_chars += len(tok)
                            unique_tokens.add(tok)

                    n_paragraphs += 1
                    if n_paragraphs % PROGRESS_EVERY == 0:
                        elapsed = time.time() - start
                        rate = n_paragraphs / elapsed
                        print(f"processed {n_paragraphs:,} paragraphs in {elapsed:.1f}s "
                              f"({rate:.0f} para/s, {len(unique_tokens):,} unique tokens so far)", flush=True)

    elapsed = time.time() - start
    print(f"DONE paragraphs={n_paragraphs:,} elapsed={elapsed:.1f}s")

    avg_sentence_length = total_words / total_sentences if total_sentences else 0
    avg_word_length = total_chars / total_words if total_words else 0
    ttr = len(unique_tokens) / total_words if total_words else 0

    lines = [
        "Corpus Statistics - IndicCorpV2 FULL guj_Gujr (all 10 parquet shards)",
        "=" * 70,
        f"Total number of sentences : {total_sentences:,}",
        f"Total number of words     : {total_words:,}",
        f"Total number of characters: {total_chars:,}",
        f"Average Sentence Length   : {avg_sentence_length:.4f} words/sentence",
        f"Average Word Length       : {avg_word_length:.4f} chars/word",
        f"Type/Token Ratio (TTR)    : {ttr:.6f}",
        f"  (unique tokens: {len(unique_tokens):,} / total tokens: {total_words:,})",
        f"Paragraphs processed      : {n_paragraphs:,}",
    ]

    with open(STATS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    stats = {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_characters": total_chars,
        "average_sentence_length": avg_sentence_length,
        "average_word_length": avg_word_length,
        "type_token_ratio": ttr,
        "unique_tokens": len(unique_tokens),
        "paragraphs_processed": n_paragraphs,
    }
    with open(STATS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
