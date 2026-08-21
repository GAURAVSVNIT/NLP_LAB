"""
Assignment-required storage format for the FULL-dataset run: stream
output_full/tokenized.jsonl.gz (23.6M tokenized sentences), join each
sentence's word tokens with spaces, and store one row per sentence in a
compressed Parquet file.

Also collects chart-ready aggregates in the same streaming pass (avoids a
second multi-minute read over the 23.6M-line file):
  - top-N most frequent "word-like" tokens (pure letters, via the
    tokenizer's own WORD_RE fullmatch - punctuation/numbers/urls excluded)
  - sentence-length (word count) distribution, bucketed at 60+

Output:
  output_full/tokenized_sentences.parquet   columns: paragraph_id, sentence
  output_full/analysis_full.json            chart-ready aggregates
"""
import gzip
import json
import sys
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import regex

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
from tokenizer import WORD_RE

WORD_FULLMATCH = regex.compile(r"^(?:" + WORD_RE[len("(?P<WORD>"):-1] + ")$")

JSONL_GZ_PATH = r"D:\CODING\NLP\Assignment_1\output_full\tokenized.jsonl.gz"
PARQUET_PATH = r"D:\CODING\NLP\Assignment_1\output_full\tokenized_sentences.parquet"
ANALYSIS_PATH = r"D:\CODING\NLP\Assignment_1\output_full\analysis_full.json"

SCHEMA = pa.schema([("paragraph_id", pa.int64()), ("sentence", pa.string())])
BATCH_SIZE = 100_000
PROGRESS_EVERY = 2_000_000
LEN_BUCKET_CAP = 60


def main():
    word_freq = Counter()
    sent_len_counts = Counter()
    n_sentences = 0

    para_ids, sentences = [], []
    start = time.time()

    def flush(writer):
        if not para_ids:
            return
        batch = pa.record_batch(
            [pa.array(para_ids, type=pa.int64()), pa.array(sentences, type=pa.string())],
            schema=SCHEMA,
        )
        writer.write_batch(batch)
        para_ids.clear()
        sentences.clear()

    with gzip.open(JSONL_GZ_PATH, "rt", encoding="utf-8") as fin, \
         pq.ParquetWriter(PARQUET_PATH, SCHEMA, compression="zstd") as writer:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj["paragraph_id"]
            for sent in obj["sentences"]:
                n = len(sent)
                bucket = n if n < LEN_BUCKET_CAP else LEN_BUCKET_CAP
                sent_len_counts[bucket] += 1
                for tok in sent:
                    if WORD_FULLMATCH.match(tok):
                        word_freq[tok] += 1

                para_ids.append(pid)
                sentences.append(" ".join(sent))
                n_sentences += 1

                if len(para_ids) >= BATCH_SIZE:
                    flush(writer)

                if n_sentences % PROGRESS_EVERY == 0:
                    elapsed = time.time() - start
                    print(f"processed {n_sentences:,} sentences in {elapsed:.1f}s "
                          f"({n_sentences/elapsed:.0f} sent/s)", flush=True)
        flush(writer)

    elapsed = time.time() - start
    print(f"DONE sentences={n_sentences:,} elapsed={elapsed:.1f}s")

    analysis = {
        "top_50_words": word_freq.most_common(50),
        "sentence_length_distribution": dict(sent_len_counts),
        "sentence_length_bucket_cap": LEN_BUCKET_CAP,
    }
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print("top 10 words:", word_freq.most_common(10))


if __name__ == "__main__":
    main()
