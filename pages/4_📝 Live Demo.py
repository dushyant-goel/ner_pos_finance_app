import streamlit as st
from utils.data_loader import load_conll_data
from utils.features import sent2features, sent2labels, sent2tokens
import pandas as pd

import sklearn_crfsuite

st.header("🧠 Live Demo: CRF for Named Entity Recognition")

st.markdown(r"""
We follow the approach used by **[Alvarado (2015)](https://aclanthology.org/U15-1010.pdf)** of using  mixed training sets for niche domains.
The model is trained on CoNLL-2003 data which is POS tagged Reuters newswire data. We divide the 
financial documents data from 8 SEC filling into two parts - first 3 documents supplment the training
and the last 5 are used to evaluate and test.           
""")

# --- Load training and test data ---
train_sents = load_conll_data("data/conll2003_train.txt")
train_sents_fin = load_conll_data("data/conll_sec_data_train.txt")

for sent in train_sents_fin:
    train_sents.append(sent)

test_sents = load_conll_data("data/conll_sec_data_test.txt")

# --- Extract features and labels ---
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# --- Train CRF ---
st.markdown("Training CRF model...")

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)
crf.fit(X_train, y_train)
st.success("CRF model trained successfully!")

# --- Save the model ---

import joblib
joblib.dump(crf, "models/crf_model.pkl")

# --- Evaluate model ---
st.subheader("📊 Model Evaluation on Test Data")

# Predict on test data
y_pred = crf.predict(X_test)
labels = list(crf.classes_)
labels.remove('O')  # Remove 'O' from labels for evaluation

import itertools 
y_test_flat = list(itertools.chain.from_iterable(y_test))
y_pred_flat = list(itertools.chain.from_iterable(y_pred))

from sklearn.metrics import classification_report
report = classification_report(y_test_flat, y_pred_flat, labels=labels)
# print(report)

# --- Convert report to df for display ---
from io import StringIO

# Read using fixed-width format
df_report = pd.read_fwf(StringIO(report), index_col=0)

# Optional: round for display
df_report = df_report.round(3)

st.dataframe(df_report.style.format(precision=2))

"---"

# --- View features of a test sentence ---
st.subheader("🔍 Explore Features for a Sentence")

sentence_idx = st.selectbox(
    "Select test sentence index",
    options=list(range(len(test_sents))),
    index=0  # default selection
)

selected = test_sents[sentence_idx]

tokens = sent2tokens(selected)
features = sent2features(selected)
predicted_labels = crf.predict_single(features)

st.markdown(f"**Sentence:** {' '.join(tokens)}")

df = pd.DataFrame({
    "Token": tokens,
    "POS": [tok[1] for tok in selected],
    "Predicted NER": predicted_labels
})
df = df.join(pd.DataFrame(features))
st.dataframe(df)

"---"

from utils.predictor import prepare_features
st.subheader("🧾 Predict Named Entities in Your Financial Document")

st.markdown("""
Paste any financial document or agreement clause below. The app will:
1. Tokenize and tag each word
2. Extract features using the CRF-compatible pipeline
3. Predict named entities (ORG, LOC, PER, MISC)
""")

user_input = st.text_area("📥 Enter your financial text here", height=200, placeholder="e.g. Lloyds Bank entered into a Credit Support Annex with Barclays Capital...")

if user_input:
    st.info("🔁 Processing your input...")

    # Load trained CRF model
    crf = joblib.load("models/crf_model.pkl")

    # Convert to CRF features
    tokens, pos_tags, features = prepare_features(user_input)

    # Predict
    y_pred = crf.predict_single(features)

    # Show result
    df = pd.DataFrame({
        "Token": tokens,
        "POS": pos_tags,
        "Predicted NER": y_pred
    })
    df = df.join(pd.DataFrame(features))
    st.dataframe(df)

    # Highlight entities
    st.subheader("🧠 Extracted Entities")
    for label in set(y_pred):
        if label != "O":
            ents = [t for t, l in zip(tokens, y_pred) if l == label]
            st.markdown(f"**{label}**: {', '.join(ents)}")
