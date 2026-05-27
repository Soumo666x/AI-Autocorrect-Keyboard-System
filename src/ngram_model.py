"""
N-gram Language Model for Next Word Prediction
Supports unigram, bigram, and trigram models with Laplace smoothing.
"""

import re
import json
import math
import pickle
from pathlib import Path
from collections import defaultdict, Counter


class NGramModel:
    """
    N-gram language model that predicts the next word based on context.
    Supports n=1 (unigram), n=2 (bigram), n=3 (trigram).
    Uses Laplace (add-1) smoothing to handle unseen n-grams.
    """

    def __init__(self, n: int = 2):
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2, or 3")
        self.n = n
        self.ngram_counts: dict = defaultdict(Counter)
        self.context_totals: dict = Counter()
        self.vocab: set = set()
        self._word_freq: Counter = Counter()   # unigram fallback counts
        self.is_trained = False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        tokens = text.split()
        return [t for t in tokens if t]

    def train(self, corpus: str | list[str]) -> None:
        """
        Train the model on a string or list of sentences.
        Multiple calls accumulate counts (incremental training).
        """
        sentences = [corpus] if isinstance(corpus, str) else corpus

        for sentence in sentences:
            tokens = self.tokenize(sentence)
            if len(tokens) < self.n:
                continue

            self.vocab.update(tokens)
            self._word_freq.update(tokens)   # unigram counts for fallback

            for i in range(len(tokens) - self.n + 1):
                if self.n == 1:
                    context = ()
                    word = tokens[i]
                elif self.n == 2:
                    context = (tokens[i],)
                    word = tokens[i + 1]
                else:  # trigram
                    context = (tokens[i], tokens[i + 1])
                    word = tokens[i + 2]

                self.ngram_counts[context][word] += 1
                self.context_totals[context] += 1

        self.is_trained = True

    def train_from_file(self, filepath: str | Path) -> None:
        """Train from a plain-text corpus file (one sentence per line)."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Corpus file not found: {filepath}")
        with open(filepath, encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        self.train(lines)
        print(f"[NGramModel] Trained on {len(lines)} sentences | "
              f"vocab={len(self.vocab)} | contexts={len(self.ngram_counts)}")

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def _build_context(self, tokens: list[str]) -> tuple:
        """Extract the relevant context tuple from a token list.

        For trigram models with fewer than 2 preceding tokens we fall back to
        a shorter context rather than returning a length-1 tuple that will
        never match any stored (length-2) key, which previously caused every
        vocabulary word to receive uniform probability (alphabetical top-k).
        """
        if self.n == 1:
            return ()
        if self.n == 2:
            return (tokens[-1],) if tokens else ()
        # trigram: need 2 preceding tokens
        if len(tokens) >= 2:
            return (tokens[-2], tokens[-1])
        # Only 1 token available → fall back to bigram-style context so that
        # we at least get frequency-weighted predictions from that word.
        if len(tokens) == 1:
            return (tokens[-1],)
        return ()

    def predict(self, text: str, top_k: int = 5) -> list[dict]:
        """
        Predict the next word(s) given partial input text.

        Returns a list of dicts:
            [{"word": str, "probability": float, "count": int}, ...]
        sorted by probability descending.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        tokens = self.tokenize(text)
        context = self._build_context(tokens)

        # Laplace-smoothed probability: P(w|context) = (C(context,w)+1)/(C(context)+V)
        vocab_size = len(self.vocab)
        candidates = self.ngram_counts.get(context, Counter())
        context_total = self.context_totals.get(context, 0)

        # ── Fallback for trigram with short input ────────────────────────────
        # If no context match exists (e.g. single token fed to a trigram model),
        # progressively shorten the context until we find counts or reach ().
        if not candidates and context:
            shorter = context[1:]           # drop oldest context token
            while shorter is not None:
                candidates = self.ngram_counts.get(shorter, Counter())
                context_total = self.context_totals.get(shorter, 0)
                if candidates:
                    break
                shorter = shorter[1:] if shorter else None

        # Last-resort fallback: return most-frequent words (unigram counts)
        if not candidates:
            top_words = [w for w, _ in self._word_freq.most_common(top_k)]
            total = sum(self._word_freq.values()) or 1
            return [
                {"word": w, "probability": self._word_freq[w] / total, "count": self._word_freq[w]}
                for w in top_words
            ]

        results = []
        all_words = self.vocab | set(candidates.keys())
        for word in all_words:
            count = candidates.get(word, 0)
            prob = (count + 1) / (context_total + vocab_size)
            results.append({"word": word, "probability": prob, "count": count})

        # Sort by probability, then alphabetically for stability
        results.sort(key=lambda x: (-x["probability"], x["word"]))
        return results[:top_k]

    def predict_words(self, text: str, top_k: int = 5) -> list[str]:
        """Convenience wrapper — returns just the word strings."""
        return [r["word"] for r in self.predict(text, top_k)]

    def perplexity(self, test_text: str) -> float:
        """
        Compute perplexity of the model on a test string.
        Lower perplexity = better model fit.
        """
        tokens = self.tokenize(test_text)
        vocab_size = len(self.vocab)
        log_prob_sum = 0.0
        count = 0

        for i in range(self.n - 1, len(tokens)):
            if self.n == 1:
                context = ()
            elif self.n == 2:
                context = (tokens[i - 1],)
            else:
                context = (tokens[i - 2], tokens[i - 1])

            word = tokens[i]
            c = self.ngram_counts.get(context, Counter()).get(word, 0)
            context_total = self.context_totals.get(context, 0)
            prob = (c + 1) / (context_total + vocab_size)
            log_prob_sum += math.log(prob)
            count += 1

        if count == 0:
            return float("inf")
        return math.exp(-log_prob_sum / count)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save the trained model to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh)
        print(f"[NGramModel] Saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load a previously saved model from a pickle file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            self.__dict__.update(pickle.load(fh))
        print(f"[NGramModel] Loaded from {path}")

    def export_json(self, path: str | Path) -> None:
        """Export model counts to a human-readable JSON file."""
        path = Path(path)
        data = {
            "n": self.n,
            "vocab_size": len(self.vocab),
            "vocab": sorted(self.vocab),
            "ngram_counts": {
                str(list(ctx)): dict(counts)
                for ctx, counts in self.ngram_counts.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"[NGramModel] Exported to {path}")

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (f"NGramModel(n={self.n}, vocab={len(self.vocab)}, "
                f"contexts={len(self.ngram_counts)}, status={status})")
