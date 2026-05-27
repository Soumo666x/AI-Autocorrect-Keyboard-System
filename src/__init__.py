"""
Autocorrect Keyboard System
===========================
A context-aware keyboard that combines next-word prediction and autocorrection.

Supported backends:
    - NGramModel  (n=1,2,3)  — fast, no dependencies beyond stdlib
    - RNNPredictor (LSTM)    — richer context, requires PyTorch
"""

from .ngram_model import NGramModel
from .autocorrect import Autocorrect
from .predictor import KeyboardPredictor

try:
    from .rnn_model import RNNPredictor
except ImportError:
    pass  # torch not installed — that's fine

__all__ = ["NGramModel", "Autocorrect", "KeyboardPredictor"]
__version__ = "1.0.0"
