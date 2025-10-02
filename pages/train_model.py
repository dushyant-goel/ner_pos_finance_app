import streamlit as st
import joblib


st.header("🧠 Live Demo: CRF for Named Entity Recognition")

st.markdown(r"""
We follow the approach used by **[Alvarado (2015)](https://aclanthology.org/U15-1010.pdf)** of using  mixed training sets for niche domains.
The model is trained on CoNLL-2003 data which is POS tagged Reuters newswire data. We divide the 
financial documents data from 8 SEC filling into two parts - first 3 documents supplement the training
and the last 5 are used to evaluate and test.           
""")

if 'model' not in st.session_state:

    # --- Load Trained CRF ---
    with st.spinner("Training CRF model..."):
        crf = joblib.load("models/crf_model.pkl")

    # --- Save the model ---
    st.session_state['model'] = crf
    st.success("CRF model trained successfully!")
else:
    st.success("CRF model trained successfully!")
