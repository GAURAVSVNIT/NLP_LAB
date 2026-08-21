"""
Extract a ~100MB partial sample of the Gujarati (guj_Gujr) text from the
downloaded IndicCorpV2 parquet shard(s) and write it as one paragraph per
line, matching the original corpus layout (data/gu.txt on HuggingFace).
"""
import sys
import pyarrow.parquet as pq

PARQUET_PATH = r"D:\CODING\NLP\Dataset\indiccorp_v2\partial-guj_Gujr\0000.parquet"
OUT_PATH = r"D:\CODING\NLP\Assignment_1\data\raw_corpus.txt"
TARGET_BYTES = 100 * 1024 * 1024  # ~100MB


def main():
    pf = pq.ParquetFile(PARQUET_PATH)
    written_bytes = 0
    written_rows = 0

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as out:
        for batch in pf.iter_batches(batch_size=10000, columns=["text"]):
            texts = batch.column("text").to_pylist()
            for t in texts:
                if t is None:
                    continue
                t = t.strip()
                if not t:
                    continue
                line = t.replace("\n", " ").replace("\r", " ") + "\n"
                out.write(line)
                written_bytes += len(line.encode("utf-8"))
                written_rows += 1
            if written_bytes >= TARGET_BYTES:
                break

    print(f"rows_written={written_rows}")
    print(f"bytes_written={written_bytes}")


if __name__ == "__main__":
    main()
