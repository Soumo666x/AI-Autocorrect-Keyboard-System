"""
RNN/LSTM Language Model for Next Word Prediction
Uses PyTorch to build a character- or word-level LSTM that predicts
the most probable next token given a sequence of preceding tokens.
"""

import re
import pickle
import math
from pathlib import Path
from collections import Counter

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────── #
#  Vocabulary
# ────────────────────────────────────────────────────────────────────────────#

class Vocabulary:
    """Maps tokens ↔ integer indices."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self._w2i: dict[str, int] = {}
        self._i2w: list[str] = []

    def build(self, token_lists: list[list[str]]) -> None:
        freq = Counter(t for seq in token_lists for t in seq)
        specials = [self.PAD, self.UNK, self.BOS, self.EOS]
        self._i2w = specials + sorted(
            w for w, c in freq.items() if c >= self.min_freq
        )
        self._w2i = {w: i for i, w in enumerate(self._i2w)}

    def encode(self, word: str) -> int:
        return self._w2i.get(word, self._w2i[self.UNK])

    def decode(self, idx: int) -> str:
        return self._i2w[idx] if 0 <= idx < len(self._i2w) else self.UNK

    def __len__(self) -> int:
        return len(self._i2w)

    @property
    def pad_idx(self) -> int:
        return self._w2i[self.PAD]

    @property
    def unk_idx(self) -> int:
        return self._w2i[self.UNK]


# ────────────────────────────────────────────────────────────────────────── #
#  Dataset
# ────────────────────────────────────────────────────────────────────────────#

_BaseDataset = Dataset if TORCH_AVAILABLE else object


class NextWordDataset(_BaseDataset):
    """
    Sliding-window dataset.
    Each sample is (context_ids, target_id) where
    context_ids = seq_len tokens and target_id = the very next token.
    """

    def __init__(self, token_lists: list[list[str]], vocab: "Vocabulary",
                 seq_len: int = 5):
        self.vocab = vocab
        self.seq_len = seq_len
        self.samples: list[tuple[list[int], int]] = []

        for tokens in token_lists:
            ids = [vocab.encode(t) for t in tokens]
            for i in range(len(ids) - seq_len):
                ctx = ids[i: i + seq_len]
                tgt = ids[i + seq_len]
                self.samples.append((ctx, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, tgt = self.samples[idx]
        if TORCH_AVAILABLE:
            return torch.tensor(ctx, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
        return ctx, tgt


# ────────────────────────────────────────────────────────────────────────── #
#  LSTM Model
# ────────────────────────────────────────────────────────────────────────────#

if TORCH_AVAILABLE:
    class LSTMLanguageModel(nn.Module):
        """
        Word-level LSTM language model.

        Architecture:
            Embedding → LSTM (stacked) → Dropout → Linear → LogSoftmax
        """

        def __init__(self, vocab_size: int, embed_dim: int = 64,
                     hidden_dim: int = 128, num_layers: int = 2,
                     dropout: float = 0.3):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embed(x)               # (B, T, E)
            out, hidden = self.lstm(emb, hidden)   # (B, T, H)
            out = self.drop(out[:, -1, :])    # last time step
            logits = self.fc(out)             # (B, V)
            return logits, hidden

        def init_hidden(self, batch_size: int, hidden_dim: int,
                        num_layers: int, device):
            return (
                torch.zeros(num_layers, batch_size, hidden_dim).to(device),
                torch.zeros(num_layers, batch_size, hidden_dim).to(device),
            )


# ────────────────────────────────────────────────────────────────────────── #
#  Trainer / Predictor wrapper
# ────────────────────────────────────────────────────────────────────────────#

class RNNPredictor:
    """
    High-level wrapper that handles training, evaluation, and inference
    for the LSTM language model.
    """

    def __init__(self, seq_len: int = 5, embed_dim: int = 64,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, min_freq: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Run: pip install torch"
            )
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.min_freq = min_freq
        self.vocab: Vocabulary | None = None
        self.model: LSTMLanguageModel | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        return [t for t in text.split() if t]

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, corpus: str | list[str], epochs: int = 10,
              batch_size: int = 32, lr: float = 1e-3,
              verbose: bool = True) -> list[float]:
        """
        Train the LSTM on the provided corpus.
        Returns a list of per-epoch average losses.
        """
        sentences = [corpus] if isinstance(corpus, str) else corpus
        token_lists = [self.tokenize(s) for s in sentences if s.strip()]

        # Build vocabulary
        self.vocab = Vocabulary(self.min_freq)
        self.vocab.build(token_lists)

        # Build dataset
        dataset = NextWordDataset(token_lists, self.vocab, self.seq_len)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty — check corpus length and seq_len.")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build model
        self.model = LSTMLanguageModel(
            len(self.vocab), self.embed_dim, self.hidden_dim,
            self.num_layers, self.dropout
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # NOTE: `verbose` was deprecated in PyTorch 2.2 and removed in 2.4.
        # We handle epoch logging ourselves below, so it is omitted here.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )

        epoch_losses: list[float] = []
        self.model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for ctx, tgt in loader:
                ctx, tgt = ctx.to(self.device), tgt.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(ctx)
                loss = criterion(logits, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            perplexity = math.exp(min(avg_loss, 700))
            epoch_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if verbose:
                print(f"  Epoch {epoch:02d}/{epochs} | "
                      f"loss={avg_loss:.4f} | ppl={perplexity:.2f}")

        self.is_trained = True
        return epoch_losses

    def train_from_file(self, filepath: str | Path, **kwargs) -> list[float]:
        """Train from a corpus file (one sentence per line)."""
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        print(f"[RNNPredictor] Training on {len(lines)} sentences …")
        return self.train(lines, **kwargs)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, text: str, top_k: int = 5,
                temperature: float = 1.0) -> list[dict]:
        """
        Predict the next word(s) given partial input.

        Args:
            text: Current input text.
            top_k: Number of suggestions to return.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            List of {"word": str, "probability": float}
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet.")

        self.model.eval()
        tokens = self.tokenize(text)

        # Pad or trim context to seq_len
        pad_idx = self.vocab.pad_idx
        ctx = [self.vocab.encode(t) for t in tokens[-self.seq_len:]]
        ctx = [pad_idx] * (self.seq_len - len(ctx)) + ctx

        with torch.no_grad():
            x = torch.tensor([ctx], dtype=torch.long).to(self.device)
            logits, _ = self.model(x)
            logits = logits[0] / temperature
            probs = torch.softmax(logits, dim=-1).cpu()

        top_probs, top_ids = probs.topk(min(top_k + 10, len(self.vocab)))
        results = []
        for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
            word = self.vocab.decode(idx)
            if word not in (Vocabulary.PAD, Vocabulary.UNK,
                            Vocabulary.BOS, Vocabulary.EOS):
                results.append({"word": word, "probability": round(prob, 6)})
            if len(results) == top_k:
                break

        return results

    def predict_words(self, text: str, top_k: int = 5,
                      temperature: float = 1.0) -> list[str]:
        """Returns just the word strings."""
        return [r["word"] for r in self.predict(text, top_k, temperature)]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "seq_len": self.seq_len,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "min_freq": self.min_freq,
            },
            "vocab": self.vocab,
            "model_state": self.model.state_dict() if self.model else None,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[RNNPredictor] Saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        cfg = payload["config"]
        self.seq_len = cfg["seq_len"]
        self.embed_dim = cfg["embed_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]
        self.min_freq = cfg["min_freq"]

        self.vocab = payload["vocab"]
        self.model = LSTMLanguageModel(
            len(self.vocab), self.embed_dim, self.hidden_dim,
            self.num_layers, self.dropout
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.is_trained = True
        print(f"[RNNPredictor] Loaded from {path}")
