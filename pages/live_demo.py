import streamlit as st
from utils.predictor import prepare_features

import pandas as pd

st.subheader("🧾 Predict Named Entities in Your Financial Document")

st.markdown("""
Paste any financial document or agreement clause below. The app will:
1. Tokenize and tag each word
2. Extract features using the CRF-compatible pipeline
3. Predict named entities (ORG, LOC, PER, MISC)
""")

if 'model' in st.session_state:
    crf = st.session_state['model']
    user_input = st.text_area("📥 Enter your financial text here", height=200, placeholder="e.g. Lloyds Bank entered into a Credit Support Annex with Barclays Capital...")
else:
    st.warning('Please first train the model.')


if user_input:
    st.info("🔁 Processing your input...")

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
