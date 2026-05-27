"""
tests/test_models.py — Unit tests for all three core modules.

Run:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.ngram_model import NGramModel
from src.autocorrect import Autocorrect
from src.predictor import KeyboardPredictor


# ── Fixtures ──────────────────────────────────────────────────────────────── #

SAMPLE_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning is transforming the world",
    "artificial intelligence powers many applications",
    "the weather today is beautiful and sunny",
    "programming requires patience and practice",
    "deep learning models require large training data",
    "natural language processing enables text understanding",
    "the project deadline is approaching fast",
]


@pytest.fixture
def bigram():
    m = NGramModel(n=2)
    m.train(SAMPLE_CORPUS)
    return m


@pytest.fixture
def trigram():
    m = NGramModel(n=3)
    m.train(SAMPLE_CORPUS)
    return m


@pytest.fixture
def ac():
    a = Autocorrect()
    a.train(SAMPLE_CORPUS)
    return a


@pytest.fixture
def predictor():
    p = KeyboardPredictor(n=2)
    p.train(SAMPLE_CORPUS)
    return p


# ── NGramModel tests ──────────────────────────────────────────────────────── #

class TestNGramModel:

    def test_train_builds_vocab(self, bigram):
        assert len(bigram.vocab) > 0

    def test_predict_returns_list(self, bigram):
        preds = bigram.predict("the", top_k=3)
        assert isinstance(preds, list)
        assert len(preds) <= 3

    def test_predict_has_probability_field(self, bigram):
        preds = bigram.predict("the", top_k=3)
        for p in preds:
            assert "word" in p
            assert "probability" in p
            assert 0 < p["probability"] <= 1

    def test_predict_words_returns_strings(self, bigram):
        words = bigram.predict_words("machine", top_k=3)
        assert all(isinstance(w, str) for w in words)

    def test_trigram_train(self, trigram):
        preds = trigram.predict("machine learning is", top_k=3)
        assert isinstance(preds, list)

    def test_unknown_word_no_crash(self, bigram):
        preds = bigram.predict("zxyqzxyzqx", top_k=3)
        assert isinstance(preds, list)

    def test_perplexity_finite(self, bigram):
        ppl = bigram.perplexity("the quick brown fox")
        assert ppl > 0
        assert ppl < float("inf")

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            NGramModel(n=5)

    def test_predict_not_trained_raises(self):
        m = NGramModel(n=2)
        with pytest.raises(RuntimeError):
            m.predict("hello")

    def test_save_load_roundtrip(self, bigram, tmp_path):
        path = tmp_path / "model.pkl"
        bigram.save(path)
        m2 = NGramModel(n=2)
        m2.load(path)
        original = bigram.predict_words("the", top_k=5)
        loaded = m2.predict_words("the", top_k=5)
        assert original == loaded


# ── Autocorrect tests ─────────────────────────────────────────────────────── #

class TestAutocorrect:

    def test_known_word_first(self, ac):
        result = ac.correct("the")
        assert result[0] == "the"

    def test_corrects_simple_typo(self, ac):
        result = ac.correct("teh", top_k=5)
        assert "the" in result

    def test_corrects_text(self, ac):
        corrected = ac.correct_text("teh weather")
        assert "the" in corrected.lower()

    def test_is_correct(self, ac):
        assert ac.is_correct("the")
        assert not ac.is_correct("xyzabcdef")

    def test_add_word(self, ac):
        ac.add_word("kubernetes")
        assert ac.is_correct("kubernetes")

    def test_vocab_size_positive(self, ac):
        assert ac.vocab_size > 0

    def test_most_common(self, ac):
        top = ac.most_common(5)
        assert len(top) <= 5
        assert all(isinstance(w, str) and isinstance(c, int) for w, c in top)


# ── KeyboardPredictor tests ───────────────────────────────────────────────── #

class TestKeyboardPredictor:

    def test_get_suggestions_structure(self, predictor):
        result = predictor.get_suggestions("the quick")
        assert "corrected_text" in result
        assert "last_word_suggestions" in result
        assert "next_word_predictions" in result
        assert "current_word" in result

    def test_next_word_predictions_are_strings(self, predictor):
        result = predictor.get_suggestions("machine")
        preds = result["next_word_predictions"]
        assert all(isinstance(w, str) for w in preds)

    def test_empty_input(self, predictor):
        result = predictor.get_suggestions("")
        assert isinstance(result["next_word_predictions"], list)
        assert result["current_word"] == ""

    def test_add_custom_word(self, predictor):
        predictor.add_user_word("tensorflow")
        assert predictor.autocorrect.is_correct("tensorflow")

    def test_stats(self, predictor):
        stats = predictor.stats()
        assert "backend" in stats
        assert "vocab_size" in stats
        assert stats["is_trained"] is True

    def test_save_load(self, predictor, tmp_path):
        path = tmp_path / "kb.pkl"
        predictor.save(path)
        p2 = KeyboardPredictor(n=2)
        p2.load(path)
        result = p2.get_suggestions("the quick")
        assert isinstance(result["next_word_predictions"], list)
