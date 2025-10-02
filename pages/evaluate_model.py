import streamlit as st

import itertools
from io import StringIO

import pandas as pd
from sklearn.metrics import classification_report

from utils.data_loader import load_conll_data
from utils.features import sent2features, sent2labels, sent2tokens
from utils.predictor import predict_on_test_data

# Load the test data set

test_sents = load_conll_data("data/conll_sec_data_test.txt")

with st.spinner('Loading test data...'):
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

if 'model' in st.session_state:
    crf = st.session_state['model']

    # --- View features of a test sentence ---
    st.subheader("🔍 Explore Features for a Sentence")

    sentence_idx = st.selectbox(
        r"Select test sentence index",
        options=list(range(len(test_sents))),
        index=21  # default selection
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

    # --- Evaluate model ---
    st.subheader("📊 Model Evaluation Statistics")

    # Predict on test data
    
    y_pred = predict_on_test_data(crf, X_test)

    labels = list(crf.classes_)
    labels.remove('O')  # Remove 'O' from labels for evaluation
    
    y_test_flat = list(itertools.chain.from_iterable(y_test))
    y_pred_flat = list(itertools.chain.from_iterable(y_pred))

    report = classification_report(y_test_flat, y_pred_flat, labels=labels)
    # print(report)

    # --- Convert report to df for display ---

    # Read using fixed-width format
    df_report = pd.read_fwf(StringIO(report), index_col=0)

    # Optional: round for display
    df_report = df_report.round(3)

    st.dataframe(df_report.style.format(precision=2))

else:
    st.warning('Please first train the model')