"""
download_datasets.py — Download and prepare larger training corpora.

Usage:
    python download_datasets.py --dataset wikitext2
    python download_datasets.py --dataset gutenberg --books 30
    python download_datasets.py --dataset all
    python download_datasets.py --list
"""

import argparse
import os
import re
import sys
import urllib.request
from pathlib import Path

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _download(url: str, dest: Path, label: str) -> None:
    """Stream-download url → dest with a simple progress bar."""
    print(f"  Downloading {label} …", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"  {size_kb:,} KB")
    except Exception as exc:
        print(f"\n  [ERROR] {exc}")
        raise


def _sentences_from_text(raw: str, min_len: int = 5) -> list[str]:
    """Split raw text into clean sentences."""
    # Remove extra whitespace and split on sentence-ending punctuation
    raw = re.sub(r"\s+", " ", raw)
    parts = re.split(r"(?<=[.!?])\s+", raw)
    return [p.strip() for p in parts if len(p.split()) >= min_len]


def _write_corpus(sentences: list[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sentences))
    print(f"  → {path}  ({len(sentences):,} sentences, "
          f"{path.stat().st_size // 1024:,} KB)")


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset 1 — WikiText-2  (~2 MB, ~36 k sentences, no extra deps)
# ─────────────────────────────────────────────────────────────────────────── #

WIKITEXT2_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/train.txt"
)


def download_wikitext2() -> None:
    print("\n[WikiText-2]")
    raw_path = OUT_DIR / "wikitext2_raw.txt"
    _download(WIKITEXT2_URL, raw_path, "WikiText-2 train split")

    raw = raw_path.read_text(encoding="utf-8")
    # Remove section headers (lines starting with ' =')
    lines = [l.strip() for l in raw.splitlines()
             if l.strip() and not l.strip().startswith("=")]
    sentences = _sentences_from_text(" ".join(lines))
    _write_corpus(sentences, OUT_DIR / "corpus_wikitext2.txt")
    raw_path.unlink()


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset 2 — Project Gutenberg classics  (plain HTTP, no API key needed)
# ─────────────────────────────────────────────────────────────────────────── #

# (id, short title)
GUTENBERG_BOOKS = [
    (1342, "pride_and_prejudice"),
    (11,   "alice_in_wonderland"),
    (84,   "frankenstein"),
    (1661, "sherlock_holmes"),
    (2701, "moby_dick"),
    (98,   "tale_of_two_cities"),
    (1400, "great_expectations"),
    (174,  "picture_of_dorian_gray"),
    (76,   "adventures_of_huckleberry_finn"),
    (1232, "the_prince"),
    (2554, "crime_and_punishment"),
    (2600, "war_and_peace"),
    (4300, "ulysses"),
    (345,  "dracula"),
    (1260, "jane_eyre"),
    (205,  "walden"),
    (16,   "peter_pan"),
    (514,  "little_women"),
    (1497, "republic"),
    (2641, "a_room_with_a_view"),
    (768,  "wuthering_heights"),
    (1080, "modest_proposal"),
    (2148, "flatland"),
    (244,  "arabian_nights"),
    (1184, "count_of_monte_cristo"),
    (2500, "siddhartha"),
    (996,  "don_quixote"),
    (55,   "wizard_of_oz"),
    (730,  "oliver_twist"),
    (36,   "war_of_the_worlds"),
]

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def download_gutenberg(n_books: int = 10) -> None:
    print(f"\n[Project Gutenberg — {n_books} books]")
    all_sentences: list[str] = []
    downloaded = 0

    for book_id, title in GUTENBERG_BOOKS[:n_books]:
        url = GUTENBERG_URL.format(id=book_id)
        tmp = OUT_DIR / f"_gut_{book_id}.txt"
        try:
            _download(url, tmp, title)
            raw = tmp.read_text(encoding="utf-8", errors="ignore")
            # Strip Gutenberg header/footer
            start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT", raw)
            end   = re.search(r"\*\*\* END OF (THE|THIS) PROJECT",   raw)
            if start:
                raw = raw[start.end():]
            if end:
                raw = raw[:end.start()]
            all_sentences.extend(_sentences_from_text(raw))
            downloaded += 1
        except Exception:
            print(f"    Skipping {title} (download failed)")
        finally:
            if tmp.exists():
                tmp.unlink()

    if all_sentences:
        _write_corpus(all_sentences, OUT_DIR / "corpus_gutenberg.txt")
    print(f"  Successfully downloaded {downloaded}/{n_books} books")


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset 3 — NLTK corpora  (Brown, Reuters, Webtext — requires nltk)
# ─────────────────────────────────────────────────────────────────────────── #

def download_nltk() -> None:
    print("\n[NLTK corpora — Brown + Reuters + Webtext]")
    try:
        import nltk
    except ImportError:
        print("  nltk not installed. Run: pip install nltk")
        return

    for pkg in ["brown", "reuters", "webtext", "punkt", "punkt_tab"]:
        nltk.download(pkg, quiet=True)

    from nltk.corpus import brown, reuters, webtext

    sentences: list[str] = []
    for corpus_obj in [brown, reuters, webtext]:
        for sent in corpus_obj.sents():
            s = " ".join(sent)
            if len(sent) >= 5:
                sentences.append(s)

    _write_corpus(sentences, OUT_DIR / "corpus_nltk.txt")


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset 4 — HuggingFace datasets  (requires `datasets` library)
# Options: wikitext, bookcorpus, openwebtext, ag_news, etc.
# ─────────────────────────────────────────────────────────────────────────── #

def download_huggingface(name: str = "wikitext",
                         config: str = "wikitext-103-raw-v1",
                         max_sentences: int = 200_000) -> None:
    print(f"\n[HuggingFace — {name}/{config}]")
    print("  Install deps: pip install datasets")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  `datasets` not installed. Run: pip install datasets")
        return

    ds = load_dataset(name, config, split="train", streaming=True,
                      trust_remote_code=True)
    sentences: list[str] = []
    for row in ds:
        text = row.get("text", "")
        if text.strip():
            sentences.extend(_sentences_from_text(text))
        if len(sentences) >= max_sentences:
            break

    out_name = f"corpus_{name.replace('/', '_')}_{config[:20]}.txt"
    _write_corpus(sentences[:max_sentences], OUT_DIR / out_name)


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset 5 — Merge all corpora into one big file
# ─────────────────────────────────────────────────────────────────────────── #

def merge_all() -> None:
    files = list(OUT_DIR.glob("corpus_*.txt"))
    if not files:
        print("\n[merge] No corpus_*.txt files found in data/")
        return
    all_lines: list[str] = []
    for f in files:
        all_lines.extend(f.read_text(encoding="utf-8").splitlines())
    # Deduplicate
    seen: set[str] = set()
    unique = [l for l in all_lines if l not in seen and not seen.add(l)]  # type: ignore[func-returns-value]
    _write_corpus(unique, OUT_DIR / "corpus.txt")
    print(f"\n[merge] Combined {len(files)} file(s) → "
          f"data/corpus.txt  ({len(unique):,} unique sentences)")


# ─────────────────────────────────────────────────────────────────────────── #
# CLI
# ─────────────────────────────────────────────────────────────────────────── #

DATASETS = {
    "wikitext2":   "~36 k sentences, ~2 MB — no dependencies",
    "gutenberg":   "Classic literature, configurable # of books — no dependencies",
    "nltk":        "Brown + Reuters + Webtext — requires: pip install nltk",
    "huggingface": "WikiText-103 or any HF dataset — requires: pip install datasets",
    "merge":       "Combine all corpus_*.txt files into data/corpus.txt",
    "all":         "Download wikitext2 + gutenberg (10 books) + nltk, then merge",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Download training corpora")
    p.add_argument("--dataset", default="wikitext2",
                   choices=list(DATASETS.keys()),
                   help="Which dataset to download")
    p.add_argument("--books", type=int, default=10,
                   help="Number of Gutenberg books (default: 10, max: 30)")
    p.add_argument("--hf-name", default="wikitext",
                   help="HuggingFace dataset name")
    p.add_argument("--hf-config", default="wikitext-103-raw-v1",
                   help="HuggingFace dataset config")
    p.add_argument("--max-sentences", type=int, default=200_000,
                   help="Max sentences for HuggingFace download")
    p.add_argument("--list", action="store_true",
                   help="List available datasets and exit")
    args = p.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name, desc in DATASETS.items():
            print(f"  {name:<14} {desc}")
        return

    ds = args.dataset
    if ds == "wikitext2":
        download_wikitext2()
    elif ds == "gutenberg":
        download_gutenberg(args.books)
    elif ds == "nltk":
        download_nltk()
    elif ds == "huggingface":
        download_huggingface(args.hf_name, args.hf_config, args.max_sentences)
    elif ds == "merge":
        merge_all()
    elif ds == "all":
        download_wikitext2()
        download_gutenberg(args.books)
        download_nltk()
        merge_all()

    print("\nDone. Train with:")
    print("  python train.py --corpus data/corpus.txt")
    print("  python train.py --corpus data/corpus.txt --n 3")
    print("  python train.py --corpus data/corpus.txt --rnn --epochs 20")


if __name__ == "__main__":
    main()
