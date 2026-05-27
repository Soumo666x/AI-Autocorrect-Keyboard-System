"""
Autocorrect Module
Detects misspelled words and suggests the closest correct alternatives
using edit-distance and phonetic heuristics.
"""

import re
from collections import Counter


def _edit1(word: str) -> set[str]:
    """Return all strings at edit-distance 1 from *word*."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:]            for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]        for L, R in splits for c in letters if R]
    inserts    = [L + c + R            for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def _edit2(word: str) -> set[str]:
    """Return all strings at edit-distance 2 from *word*."""
    return {e2 for e1 in _edit1(word) for e2 in _edit1(e1)}


class Autocorrect:
    """
    Autocorrect engine backed by a frequency dictionary.

    Usage:
        ac = Autocorrect()
        ac.train(corpus_text)
        corrections = ac.correct("teh")   # → ["the"]
    """

    def __init__(self):
        self._word_freq: Counter = Counter()
        self._vocab: set = set()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, text: str | list[str]) -> None:
        """Build word-frequency dictionary from raw text or list of strings."""
        if isinstance(text, list):
            text = " ".join(text)
        words = re.findall(r"[a-z']+", text.lower())
        self._word_freq.update(words)
        self._vocab = set(self._word_freq.keys())

    def train_from_file(self, filepath: str) -> None:
        with open(filepath, encoding="utf-8") as fh:
            self.train(fh.read())

    # ------------------------------------------------------------------ #
    # Core correction logic
    # ------------------------------------------------------------------ #

    def _candidates(self, word: str) -> list[str]:
        """
        Return correction candidates in priority order:
        1. The word itself (if known)
        2. Edit-distance-1 known words
        3. Edit-distance-2 known words
        4. The word itself (fallback)
        """
        known_self  = self._known([word])
        known_edit1 = self._known(_edit1(word))
        known_edit2 = self._known(_edit2(word))

        pool = known_self or known_edit1 or known_edit2 or {word}
        # Sort by frequency descending, then alphabetically for stability
        return sorted(pool, key=lambda w: (-self._word_freq[w], w))

    def _known(self, words) -> set[str]:
        return {w for w in words if w in self._vocab}

    def correct(self, word: str, top_k: int = 5) -> list[str]:
        """
        Return the top-k most likely corrections for a potentially misspelled word.
        If the word is in the vocabulary, it is returned unchanged as the first entry.
        """
        word = word.lower().strip()
        candidates = self._candidates(word)
        return candidates[:top_k]

    def correct_text(self, text: str) -> str:
        """
        Attempt to correct every word in a sentence.
        Only replaces words not found in the vocabulary.
        Preserves original capitalization where possible.
        """
        tokens = text.split()
        corrected = []
        for token in tokens:
            clean = re.sub(r"[^a-z']", "", token.lower())
            if not clean:
                corrected.append(token)
                continue
            best = self.correct(clean, top_k=1)
            suggestion = best[0] if best else clean
            # Restore leading/trailing punctuation
            prefix = re.match(r"^[^a-zA-Z]*", token).group()
            suffix = re.search(r"[^a-zA-Z]*$", token).group()
            corrected.append(prefix + suggestion + suffix)
        return " ".join(corrected)

    def is_correct(self, word: str) -> bool:
        """Return True if the word exists in the vocabulary."""
        return word.lower() in self._vocab

    def add_word(self, word: str, freq: int = 10) -> None:
        """Manually add a word to the vocabulary (e.g. user-defined terms)."""
        word = word.lower()
        self._word_freq[word] += freq
        self._vocab.add(word)

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def most_common(self, n: int = 20) -> list[tuple[str, int]]:
        return self._word_freq.most_common(n)
