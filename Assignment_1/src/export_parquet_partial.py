"""
Assignment-required storage format for the partial-sample run:
tokenize each paragraph into sentences, join each sentence's word tokens
with spaces, and store one row per tokenized sentence in a compressed
Parquet file (instead of a giant plain-text file).

Also collects the data needed for the visual analysis charts:
  - token-type distribution (WORD / NUMBER / PUNCT / URL / EMAIL / DATE / ...)
  - top-N most frequent WORD tokens
  - sentence-length (word count) distribution

Output:
  output/tokenized_sentences.parquet   columns: paragraph_id, sentence
  output/analysis_partial.json         chart-ready aggregates
"""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "src")
from tokenizer import sentence_tokenize

RAW_PATH = r"D:\CODING\NLP\Assignment_1\data\raw_corpus.txt"
PARQUET_PATH = r"D:\CODING\NLP\Assignment_1\output\tokenized_sentences.parquet"
ANALYSIS_PATH = r"D:\CODING\NLP\Assignment_1\output\analysis_partial.json"

SCHEMA = pa.schema([("paragraph_id", pa.int64()), ("sentence", pa.string())])
BATCH_SIZE = 50_000


def main():
    type_counts = Counter()
    word_freq = Counter()
    sent_len_counts = Counter()

    para_ids, sentences = [], []

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

    with open(RAW_PATH, "r", encoding="utf-8") as fin, \
         pq.ParquetWriter(PARQUET_PATH, SCHEMA, compression="zstd") as writer:

        for pid, line in enumerate(fin):
            para = line.strip()
            if not para:
                continue
            for sent in sentence_tokenize(para):
                words = [tok.text for tok in sent]
                sent_len_counts[len(words)] += 1
                for tok in sent:
                    type_counts[tok.type] += 1
                    if tok.type == "WORD":
                        word_freq[tok.text] += 1

                para_ids.append(pid)
                sentences.append(" ".join(words))
                if len(para_ids) >= BATCH_SIZE:
                    flush(writer)
        flush(writer)

    analysis = {
        "token_type_distribution": dict(type_counts),
        "top_50_words": word_freq.most_common(50),
        "sentence_length_distribution": dict(sent_len_counts),
    }
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print("token_type_distribution:", dict(type_counts))
    print("top 10 words:", word_freq.most_common(10))


if __name__ == "__main__":
    main()
