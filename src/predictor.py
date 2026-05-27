"""
Keyboard Predictor
Unified interface that combines autocorrect and next-word prediction.
Supports both NGramModel and RNNPredictor backends.
"""

from __future__ import annotations
import pickle
import re
import warnings
from pathlib import Path

from .autocorrect import Autocorrect
from .ngram_model import NGramModel


class KeyboardPredictor:
    """
    Unified keyboard prediction engine.

    Combines:
    - Autocorrect:   fixes misspelled words as the user types
    - Next-word:     suggests the most likely next word(s)

    By default uses a bigram (n=2) NGramModel. Swap in RNNPredictor for
    a deep-learning backend (requires PyTorch).

    Example usage:
        predictor = KeyboardPredictor()
        predictor.train_from_file("data/corpus.txt")
        predictor.get_suggestions("the quick brown")
    """

    def __init__(self, n: int = 2, use_rnn: bool = False,
                 rnn_kwargs: dict | None = None):
        """
        Args:
            n:          N-gram order (1, 2, or 3). Ignored if use_rnn=True.
            use_rnn:    Use LSTM-based predictor instead of n-gram.
            rnn_kwargs: Extra kwargs forwarded to RNNPredictor.__init__.
        """
        self.autocorrect = Autocorrect()

        if use_rnn:
            from .rnn_model import RNNPredictor
            self._predictor = RNNPredictor(**(rnn_kwargs or {}))
        else:
            self._predictor = NGramModel(n=n)

        self._use_rnn = use_rnn
        self.is_trained = False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, corpus: str | list[str], **kwargs) -> None:
        """Train both autocorrect and prediction model on the same corpus."""
        text = corpus if isinstance(corpus, str) else "\n".join(corpus)
        self.autocorrect.train(text)
        if self._use_rnn:
            self._predictor.train(corpus, **kwargs)
        else:
            self._predictor.train(corpus)
        self.is_trained = True

    def train_from_file(self, filepath: str | Path, **kwargs) -> None:
        """Load corpus from file and train both models."""
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        self.train(lines, **kwargs)
        print(f"[KeyboardPredictor] Trained on {len(lines)} sentences")

    # ------------------------------------------------------------------ #
    # Suggestions
    # ------------------------------------------------------------------ #

    def get_suggestions(self, text: str, top_k: int = 5,
                        correct_last: bool = True) -> dict:
        """
        Given the current input text, return autocorrect and prediction results.

        Args:
            text:         Text the user has typed so far.
            top_k:        Number of next-word suggestions.
            correct_last: Whether to autocorrect the last (possibly incomplete) word.

        Returns a dict:
        {
            "corrected_text": str,          # full text after autocorrect
            "last_word_suggestions": [...], # corrections for the last word
            "next_word_predictions": [...], # predicted next words
            "current_word": str,
        }
        """
        tokens = re.findall(r"[a-z']+", text.lower())
        current_word = tokens[-1] if tokens else ""

        # Autocorrect last word only
        last_word_suggestions = []
        corrected_text = text
        if correct_last and current_word and len(current_word) >= 2:
            last_word_suggestions = self.autocorrect.correct(current_word, top_k)
            # Only show if the top suggestion differs (actual typo)
            if last_word_suggestions and last_word_suggestions[0] != current_word:
                # Replace last word in text with corrected version
                pattern = re.compile(re.escape(current_word) + r"$", re.IGNORECASE)
                corrected_text = pattern.sub(last_word_suggestions[0], text.rstrip())

        # Next-word predictions
        next_word_predictions: list[str] = []
        if self.is_trained:
            try:
                next_word_predictions = self._predictor.predict_words(
                    corrected_text, top_k=top_k
                )
            except RuntimeError:
                # Model not trained yet (e.g. freshly loaded partial state)
                next_word_predictions = []
            except Exception as exc:           # unexpected – surface it as a warning
                warnings.warn(f"[KeyboardPredictor] predict_words failed: {exc}",
                              RuntimeWarning, stacklevel=2)
                next_word_predictions = []

        return {
            "corrected_text": corrected_text,
            "last_word_suggestions": last_word_suggestions,
            "next_word_predictions": next_word_predictions,
            "current_word": current_word,
        }

    def add_user_word(self, word: str, freq: int = 10) -> None:
        """Teach the predictor a custom word (e.g. a name or technical term)."""
        self.autocorrect.add_word(word, freq)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    # File format (v2): a single pickle that stores both the language-model
    # object and the Autocorrect word-frequency table so that a save→load
    # round-trip produces identical behaviour.
    #
    # Backward compat: if the file was written by v1 (only the raw
    # NGramModel/__dict__ was pickled) we fall back gracefully — the
    # autocorrect vocab will be empty but prediction still works.

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_format_version": 2,
            "use_rnn": self._use_rnn,
            # Serialise the entire predictor object (NGramModel or RNNPredictor)
            "predictor": self._predictor,
            # Autocorrect: store the frequency counter and vocab set
            "autocorrect_word_freq": dict(self.autocorrect._word_freq),
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[KeyboardPredictor] Saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        if isinstance(payload, dict) and payload.get("_format_version") == 2:
            # ── New v2 format ────────────────────────────────────────────
            from collections import Counter
            self._predictor = payload["predictor"]
            self._use_rnn = payload.get("use_rnn", self._use_rnn)
            freq = payload.get("autocorrect_word_freq", {})
            self.autocorrect._word_freq = Counter(freq)
            self.autocorrect._vocab = set(freq.keys())
        else:
            # ── Legacy v1 format (raw NGramModel.__dict__) ───────────────
            warnings.warn(
                "Loading a v1 model file — autocorrect vocab will be empty. "
                "Re-train and re-save to upgrade to v2 format.",
                UserWarning,
                stacklevel=2,
            )
            self._predictor.__dict__.update(payload)

        self.is_trained = True
        print(f"[KeyboardPredictor] Loaded from {path}")

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "backend": "rnn" if self._use_rnn else f"{self._predictor.n}-gram",
            "vocab_size": self.autocorrect.vocab_size,
            "is_trained": self.is_trained,
            "top_words": self.autocorrect.most_common(10),
        }
