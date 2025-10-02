import streamlit as st

from nltk import word_tokenize, pos_tag
from utils.features import sent2features

@st.cache_data
def tokenize_and_tag(text):
    
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    # Convert to dummy NER-tag format expected: (word, POS, _)
    return [(word, pos, "O") for word, pos in tagged]

@st.cache_data
def prepare_features(text):
    tagged_sent = tokenize_and_tag(text)
    features = sent2features(tagged_sent)
    tokens = [w for w, p, t in tagged_sent]
    pos_tags = [p for w, p, t in tagged_sent]
    return tokens, pos_tags, features

@st.cache_data
def predict_on_test_data(_crf_model, X_test):
    y_pred = _crf_model.predict(X_test)
    return y_pred