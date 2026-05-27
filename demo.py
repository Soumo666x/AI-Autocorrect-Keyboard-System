"""
demo.py — Interactive CLI demo of the autocorrect keyboard system.

Usage:
    python demo.py                  # uses built-in bigram model
    python demo.py --n 3            # trigram
    python demo.py --model models/keyboard_model.pkl  # load pre-trained
    python demo.py --rnn            # LSTM (requires PyTorch)

Type text at the prompt; the system prints suggestions live.
Type :quit or Ctrl-C to exit.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.predictor import KeyboardPredictor


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          Autocorrect Keyboard — Interactive Demo             ║
║  Type text and get live predictions + autocorrect            ║
║  Commands: :quit  :clear  :stats  :add <word>                ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autocorrect Keyboard CLI demo")
    p.add_argument("--n", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--rnn", action="store_true")
    p.add_argument("--corpus", default="data/corpus.txt")
    p.add_argument("--model", default=None,
                   help="Path to a pre-saved model (.pkl); skips training")
    return p.parse_args()


def render_result(result: dict, top_k: int = 5) -> None:
    cw   = result["current_word"]
    ac   = result["last_word_suggestions"]
    pred = result["next_word_predictions"]

    if ac and ac[0] != cw and cw:
        print(f"\n  ✏️  Autocorrect  '{cw}'  →  {ac[:top_k]}")

    if pred:
        bar = "  ".join(f"[{w}]" for w in pred[:top_k])
        print(f"  📋 Next word   →  {bar}")
    else:
        print("  (no predictions yet)")


def main() -> None:
    args = parse_args()
    print(BANNER)

    predictor = KeyboardPredictor(n=args.n, use_rnn=args.rnn)

    if args.model and Path(args.model).exists():
        print(f"Loading model from {args.model} …")
        predictor.load(args.model)
    else:
        print(f"Training on {args.corpus} …")
        try:
            predictor.train_from_file(args.corpus)
        except FileNotFoundError:
            print(f"[ERROR] Corpus not found: {args.corpus}", file=sys.stderr)
            sys.exit(1)

    print("\nReady!  Start typing below.\n")
    buffer = ""

    while True:
        try:
            chunk = input(f"  > {buffer}")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        # Commands
        stripped = chunk.strip()
        if stripped == ":quit":
            print("Bye!")
            break
        elif stripped == ":clear":
            buffer = ""
            print("  (buffer cleared)")
            continue
        elif stripped == ":stats":
            import json
            print(json.dumps(predictor.stats(), indent=2, default=str))
            continue
        elif stripped.startswith(":add "):
            word = stripped[5:].strip()
            if word:
                predictor.add_user_word(word)
                print(f"  Added '{word}' to vocabulary.")
            continue

        buffer = (buffer + " " + chunk).strip()
        result = predictor.get_suggestions(buffer, top_k=5)
        render_result(result)
        print()


if __name__ == "__main__":
    main()
