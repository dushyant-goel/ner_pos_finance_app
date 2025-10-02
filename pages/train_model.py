import streamlit as st

from utils.data_loader import load_conll_data
from utils.features import sent2features, sent2labels

import sklearn_crfsuite

st.header("🧠 Live Demo: CRF for Named Entity Recognition")

st.markdown(r"""
We follow the approach used by **[Alvarado (2015)](https://aclanthology.org/U15-1010.pdf)** of using  mixed training sets for niche domains.
The model is trained on CoNLL-2003 data which is POS tagged Reuters newswire data. We divide the 
financial documents data from 8 SEC filling into two parts - first 3 documents supplement the training
and the last 5 are used to evaluate and test.           
""")

if 'model' not in st.session_state:
    # --- Load training and test data ---
    train_sents = load_conll_data("data/conll2003_train.txt")
    train_sents_fin = load_conll_data("data/conll_sec_data_train.txt")

    train_sents.extend(train_sents_fin)

    # --- Extract features and labels ---
    with st.spinner('Extracting features for training...'):
        X_train = [sent2features(s) for s in train_sents]
        y_train = [sent2labels(s) for s in train_sents]

    # --- Train CRF ---
    with st.spinner("Training CRF model..."):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )
        crf.fit(X_train, y_train)

    # --- Save the model ---
    st.session_state['model'] = crf
    st.success("CRF model trained successfully!")
else:
    st.success("CRF model trained successfully!")
