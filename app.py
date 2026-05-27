"""
app.py — Streamlit web UI for the autocorrect keyboard system.

Run:
    pip install streamlit
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.predictor import KeyboardPredictor

# ── Page config ──────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Autocorrect Keyboard",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.prediction-btn {
    display: inline-block;
    background: #f0f2f6;
    border-radius: 8px;
    padding: 6px 14px;
    margin: 4px;
    font-size: 14px;
    cursor: pointer;
    border: 1px solid #ddd;
}
.highlight { background-color: #fff3cd; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────── #
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar: configuration ─────────────────────────────────────────────────── #
with st.sidebar:
    st.title("⌨️ Settings")

    backend = st.radio("Prediction backend", ["N-gram (fast)", "LSTM / RNN"],
                       help="LSTM requires PyTorch to be installed.")

    if backend == "N-gram (fast)":
        n_order = st.select_slider("N-gram order", options=[1, 2, 3], value=2,
                                   help="2 = bigram (best for small corpora)")
    else:
        n_order = 2  # unused

    top_k = st.slider("Suggestions count", 1, 8, 5)

    corpus_path = st.text_input("Corpus path", value="data/corpus.txt")

    if st.button("🚀 Train / Reload model", use_container_width=True):
        use_rnn = backend == "LSTM / RNN"
        with st.spinner("Training…"):
            predictor = KeyboardPredictor(n=n_order, use_rnn=use_rnn)
            try:
                predictor.train_from_file(corpus_path)
                st.session_state.predictor = predictor
                st.success(f"Model trained! Vocab size: {predictor.stats()['vocab_size']}")
            except FileNotFoundError:
                st.error(f"Corpus not found: {corpus_path}")
            except ImportError as e:
                st.error(f"Missing dependency: {e}")

    st.divider()
    st.markdown("**Add custom word**")
    custom_word = st.text_input("Word", placeholder="e.g. GPT, numpy…")
    if st.button("Add word") and custom_word:
        if st.session_state.predictor:
            st.session_state.predictor.add_user_word(custom_word)
            st.success(f"Added: {custom_word}")
        else:
            st.warning("Train the model first.")


# ── Main area ─────────────────────────────────────────────────────────────── #
st.title("Autocorrect Keyboard System")
st.caption("Start typing — the keyboard predicts the next word and corrects typos in real time.")

if st.session_state.predictor is None:
    st.info("⬅  Click **Train / Reload model** in the sidebar to get started.")
else:
    predictor: KeyboardPredictor = st.session_state.predictor

    # ── Input ──────────────────────────────────────────────────────────────── #
    col1, col2 = st.columns([3, 1])
    with col1:
        user_text = st.text_area(
            "Your text",
            value=st.session_state.input_text,
            height=120,
            placeholder="Start typing here…",
            key="text_area",
        )
        st.session_state.input_text = user_text
    with col2:
        st.metric("Characters", len(user_text))
        words = [w for w in user_text.split() if w]
        st.metric("Words", len(words))
        st.metric("Sentences",
                  sum(1 for c in user_text if c in ".!?") or (1 if user_text.strip() else 0))

    # ── Suggestions ────────────────────────────────────────────────────────── #
    if user_text.strip():
        result = predictor.get_suggestions(user_text, top_k=top_k)
        st.divider()

        # Autocorrect
        if (result["last_word_suggestions"] and
                result["last_word_suggestions"][0] != result["current_word"] and
                result["current_word"]):
            st.markdown("**Autocorrect suggestions for** "
                        f"`{result['current_word']}`")
            ac_cols = st.columns(min(len(result["last_word_suggestions"]), 5))
            for i, word in enumerate(result["last_word_suggestions"][:5]):
                with ac_cols[i]:
                    if st.button(f"✏️ {word}", key=f"ac_{i}_{word}",
                                 use_container_width=True):
                        import re
                        pattern = re.compile(
                            re.escape(result["current_word"]) + r"$", re.IGNORECASE
                        )
                        st.session_state.input_text = pattern.sub(
                            word, user_text.rstrip()
                        ) + " "
                        st.rerun()

        # Next-word predictions
        if result["next_word_predictions"]:
            st.markdown("**Predicted next words**")
            pred_cols = st.columns(min(len(result["next_word_predictions"]), 5))
            for i, word in enumerate(result["next_word_predictions"][:5]):
                with pred_cols[i]:
                    if st.button(f"➕ {word}", key=f"nw_{i}_{word}",
                                 use_container_width=True):
                        st.session_state.input_text = user_text.rstrip() + f" {word} "
                        st.rerun()
        else:
            st.info("No next-word predictions for this input yet.")

    # ── History ────────────────────────────────────────────────────────────── #
    st.divider()
    with st.expander("Typing history"):
        if st.button("Save current text to history"):
            if user_text.strip():
                st.session_state.history.append(user_text.strip())
        if st.button("Clear history"):
            st.session_state.history = []
        for i, entry in enumerate(reversed(st.session_state.history[-10:])):
            st.caption(f"{i+1}. {entry}")

    # ── Model stats ─────────────────────────────────────────────────────────── #
    with st.expander("Model statistics"):
        stats = predictor.stats()
        st.json(stats)
