import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import pickle
import json

# Resolve model path relative to this file and support .h5 or SavedModel folder
app_dir = os.path.dirname(__file__)
h5_path = os.path.join(app_dir, "fake_news_model.h5")
saved_model_dir = os.path.join(app_dir, "saved_model")  # adjust if needed
tokenizer_pickle = os.path.join(app_dir, "tokenizer.pickle")
tokenizer_json = os.path.join(app_dir, "tokenizer.json")

# Load model
try:
    if os.path.exists(h5_path):
        model = load_model(h5_path)
    elif os.path.isdir(saved_model_dir):
        model = load_model(saved_model_dir)
    else:
        raise FileNotFoundError("No 'fake_news_model.h5' file or 'saved_model' directory found in the app folder.")
except Exception as e:
    st.title("📰 Fake News Classifier")
    st.error(
        "Could not load model.\nPlace 'fake_news_model.h5' or a SavedModel folder named 'saved_model' inside the app folder.\n\n"
        f"Load error: {e}"
    )
    st.stop()

# Infer embedding params
embedding_layer = None
for layer in model.layers:
    if isinstance(layer, Embedding):
        embedding_layer = layer
        break

if embedding_layer is not None:
    voc_size = int(getattr(embedding_layer, "input_dim", 5000))
    sent_length = getattr(embedding_layer, "input_length", None)
    if not sent_length:
        try:
            sent_length = int(model.input_shape[1])
        except Exception:
            sent_length = 20
else:
    voc_size = 5000
    sent_length = 20

# Try to load a saved tokenizer (optional). If none, we'll use one_hot fallback.
tokenizer = None
if os.path.exists(tokenizer_pickle):
    try:
        with open(tokenizer_pickle, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception:
        tokenizer = None
elif os.path.exists(tokenizer_json):
    try:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        with open(tokenizer_json, "r", encoding="utf-8") as f:
            tokenizer = tokenizer_from_json(json.load(f))
    except Exception:
        tokenizer = None

st.title("📰 Fake News Classifier")
st.write("This app predicts whether a news article is **Fake** or **Real** using a Bidirectional LSTM model.")

# User input
user_input = st.text_area("Enter the news text to analyze:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        try:
            # Use saved tokenizer if available (preserves training indices)
            if tokenizer is not None:
                seqs = tokenizer.texts_to_sequences([user_input])
            else:
                # fallback: one_hot mapping respecting current voc_size
                seqs = [one_hot(user_input, voc_size)]

            # pad and ensure integer dtype
            X = pad_sequences(seqs, padding='pre', maxlen=sent_length)
            X = np.asarray(X, dtype=np.int32)

            # Clip to valid range for the model's Embedding layer to avoid out-of-range indices
            X = np.clip(X, 0, voc_size - 1)

            # Predict
            pred = model.predict(X)
            # handle different output shapes
            score = float(pred.ravel()[0])

            if score > 0.5:
                st.success(f"✅ The news is likely REAL (score={score:.3f})")
            else:
                st.error(f"🚨 The news is likely FAKE (score={score:.3f})")

        except AttributeError as ae:
            # catch "'NoneType' object has no attribute 'pop'" and similar attribute errors
            st.error(
                "An internal preprocessing object is missing (None). "
                "If you trained with a Tokenizer, ensure 'tokenizer.pickle' or 'tokenizer.json' is in the app folder. "
                f"\n\nAttributeError: {ae}"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.write("---")
st.caption("Built with Streamlit and TensorFlow by Srajal Tiwari")
