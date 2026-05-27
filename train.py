"""
train.py — Train and save an autocorrect keyboard model.

Usage:
    python train.py                          # bigram model, default corpus
    python train.py --n 3                    # trigram model
    python train.py --rnn --epochs 20        # LSTM model (requires PyTorch)
    python train.py --corpus data/corpus.txt --out models/my_model.pkl
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from src.predictor import KeyboardPredictor
from src.ngram_model import NGramModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the autocorrect keyboard model")
    p.add_argument("--corpus", default="data/corpus.txt",
                   help="Path to training corpus (one sentence per line)")
    p.add_argument("--out", default="models/keyboard_model.pkl",
                   help="Where to save the trained model")
    p.add_argument("--n", type=int, default=2, choices=[1, 2, 3],
                   help="N-gram order (ignored if --rnn is set)")
    p.add_argument("--rnn", action="store_true",
                   help="Use LSTM model instead of n-gram (requires PyTorch)")
    p.add_argument("--epochs", type=int, default=15,
                   help="Training epochs (RNN only)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size (RNN only)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (RNN only)")
    p.add_argument("--export-json", action="store_true",
                   help="Also export n-gram counts as JSON")
    p.add_argument("--eval", action="store_true",
                   help="Run a quick evaluation after training")
    return p.parse_args()


def quick_eval(predictor: KeyboardPredictor) -> None:
    """Print a few example predictions to eyeball model quality."""
    test_prompts = [
        "the quick",
        "machine learning",
        "artificial intelligence",
        "the weather",
        "deep learning",
    ]
    print("\n── Quick evaluation ──────────────────────────────────────")
    for prompt in test_prompts:
        result = predictor.get_suggestions(prompt, top_k=5)
        preds = result["next_word_predictions"]
        print(f"  '{prompt}' → {preds}")
    print("──────────────────────────────────────────────────────────\n")


def main() -> None:
    args = parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[ERROR] Corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[train] Backend  : {'LSTM/RNN' if args.rnn else f'{args.n}-gram'}")
    print(f"[train] Corpus   : {corpus_path}")
    print(f"[train] Output   : {args.out}")

    # Build predictor
    predictor = KeyboardPredictor(n=args.n, use_rnn=args.rnn)

    # Train
    rnn_kwargs = {"epochs": args.epochs, "batch_size": args.batch_size,
                  "lr": args.lr}
    try:
        predictor.train_from_file(
            corpus_path, **(rnn_kwargs if args.rnn else {})
        )
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}", file=sys.stderr)
        raise

    # Save model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(out_path)

    # Optional JSON export (n-gram only)
    if args.export_json and not args.rnn:
        json_path = out_path.with_suffix(".json")
        predictor._predictor.export_json(json_path)

    # Stats
    stats = predictor.stats()
    print(f"\n[train] Stats: {json.dumps(stats, indent=2, default=str)}")

    # Eval
    if args.eval:
        quick_eval(predictor)

    print("[train] Done ✓")


if __name__ == "__main__":
    main()
