"""
Compute corpus statistics from output/tokenized.jsonl:
  i.   Total number of sentences
  ii.  Total number of words (tokens)
  iii. Total number of characters (sum of character lengths of all word tokens)
  iv.  Average Sentence Length (avg number of words per sentence)
  v.   Average Word Length (avg number of characters per word)
  vi.  Type/Token Ratio (unique tokens / total tokens)

"Words" here = every token produced by the word tokenizer (word, number,
date, email, url and punctuation tokens alike), since the assignment asks
the tokenizer itself to treat punctuation/urls/numbers/dates/emails as
tokens. TTR and character counts use the same token stream for consistency.
"""
import json

JSONL_PATH = r"D:\CODING\NLP\Assignment_1\output\tokenized.jsonl"
STATS_TXT_PATH = r"D:\CODING\NLP\Assignment_1\output\corpus_statistics.txt"
STATS_JSON_PATH = r"D:\CODING\NLP\Assignment_1\output\corpus_statistics.json"


def main():
    total_sentences = 0
    total_words = 0
    total_chars = 0
    unique_tokens = set()

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for sent in obj["sentences"]:
                total_sentences += 1
                total_words += len(sent)
                for tok in sent:
                    total_chars += len(tok)
                    unique_tokens.add(tok)

    avg_sentence_length = total_words / total_sentences if total_sentences else 0
    avg_word_length = total_chars / total_words if total_words else 0
    ttr = len(unique_tokens) / total_words if total_words else 0

    lines = [
        "Corpus Statistics - IndicCorpV2 partial-guj_Gujr",
        "=" * 50,
        f"Total number of sentences : {total_sentences:,}",
        f"Total number of words     : {total_words:,}",
        f"Total number of characters: {total_chars:,}",
        f"Average Sentence Length   : {avg_sentence_length:.4f} words/sentence",
        f"Average Word Length       : {avg_word_length:.4f} chars/word",
        f"Type/Token Ratio (TTR)    : {ttr:.6f}",
        f"  (unique tokens: {len(unique_tokens):,} / total tokens: {total_words:,})",
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
    }
    with open(STATS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
