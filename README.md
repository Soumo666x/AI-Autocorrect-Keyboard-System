# Autocorrect Keyboard System

A context-aware keyboard that **predicts the next word** and **autocorrects typos** using n-gram language models and an optional LSTM/RNN backend.

---

## Features

| Feature | Detail |
|---|---|
| **Next-word prediction** | Bigram / trigram n-gram model with Laplace smoothing |
| **LSTM / RNN prediction** | Word-level stacked LSTM trained with PyTorch |
| **Autocorrect** | Edit-distance spell corrector (Norvig algorithm) |
| **Web UI** | Streamlit app with live suggestions |
| **CLI demo** | Interactive terminal session |
| **Extensible** | Swap in any text corpus; add custom words at runtime |

---

## Project Structure

```
autocorrect_keyboard/
├── app.py                  # Streamlit web application
├── demo.py                 # Interactive CLI demo
├── train.py                # Training script
├── requirements.txt
├── data/
│   └── corpus.txt          # Default training corpus
├── src/
│   ├── __init__.py
│   ├── ngram_model.py      # N-gram language model
│   ├── rnn_model.py        # LSTM language model (PyTorch)
│   ├── autocorrect.py      # Edit-distance autocorrect engine
│   └── predictor.py        # Unified KeyboardPredictor interface
├── models/                 # Saved model files (created at runtime)
└── tests/
    └── test_models.py
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/autocorrect-keyboard.git
cd autocorrect-keyboard
pip install -r requirements.txt
```

### 2. Launch the web app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### 3. CLI demo

```bash
python demo.py
```

Type freely. The system shows predictions after every line.

---

## Training

### N-gram model (default)

```bash
# Bigram — fast, works without any extra dependencies
python train.py

# Trigram — more context, slightly slower
python train.py --n 3

# Custom corpus
python train.py --corpus path/to/mytext.txt --out models/custom.pkl

# Evaluate after training
python train.py --eval
```

### LSTM / RNN model

> Requires PyTorch: `pip install torch`

```bash
python train.py --rnn --epochs 20 --lr 1e-3
```

---

## Python API

```python
from src.predictor import KeyboardPredictor

# Create and train
predictor = KeyboardPredictor(n=2)          # or use_rnn=True
predictor.train_from_file("data/corpus.txt")

# Get suggestions
result = predictor.get_suggestions("the quick brown")
print(result["next_word_predictions"])   # e.g. ['fox', 'dog', 'bird']
print(result["last_word_suggestions"])   # autocorrect of last word

# Add a custom word (names, jargon, etc.)
predictor.add_user_word("tensorflow")

# Save / load
predictor.save("models/my_model.pkl")
predictor.load("models/my_model.pkl")
```

---

## How It Works

### N-gram Model

An **n-gram** is a contiguous sequence of *n* words. For a bigram (n=2):

```
"the quick brown fox"
 → ("the","quick"), ("quick","brown"), ("brown","fox")
```

Given the last word typed, the model counts how often each word follows it in the training corpus and returns the top-k most frequent successors.  
**Laplace smoothing** (add-1) ensures unseen words receive non-zero probability.

### LSTM Model

A word-level **Long Short-Term Memory** network that:
1. Maps each word to a dense embedding vector
2. Passes the sequence through stacked LSTM layers
3. Predicts a probability distribution over the vocabulary for the next token

This captures **longer-range context** than n-grams and handles polysemy better.

### Autocorrect

Implements the **Norvig spell corrector**:
1. Generate all strings at edit-distance 1 (deletions, transpositions, replacements, insertions)
2. If none are known words, generate edit-distance 2 candidates
3. Rank by corpus frequency

---

## Running Tests

```bash
python -m pytest tests/ -v
```

To also see coverage:

```bash
pip install pytest-cov
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Extending the Project

### Use your own corpus

Replace `data/corpus.txt` with any plain-text file (one sentence per line works best):

```bash
python train.py --corpus data/my_corpus.txt --out models/custom_model.pkl
```

### Improve predictions

- Use a **larger corpus** (Wikipedia dumps, book datasets, domain-specific text)
- Increase **n** (trigrams capture more context)
- For deep learning: increase `--epochs`, add more LSTM layers, or use a pre-trained embedding

---

## Requirements

| Requirement | Version |
|---|---|
| Python | >= 3.10 |
| streamlit | >= 1.32 |
| torch *(optional)* | >= 2.0 |
| pytest | >= 7.4 |

---

## License

MIT License — see `LICENSE` for details.
