import streamlit as st

from nltk.stem import PorterStemmer, WordNetLemmatizer
import string


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

@st.cache_data
def word_shape(word):
    return ''.join(
        'X' if c.isupper() else
        'x' if c.islower() else
        '0' if c.isdigit() else
        '#' for c in word
    )

@st.cache_data
def word2features(sent, i):
    word, postag = sent[i][0], sent[i][1]
    return {
        'word': word.lower(),
        'is_capitalized': word[0].isupper(),
        'has_capitals_inside': any(c.isupper() for c in word[1:]),
        'has_digit': any(c.isdigit() for c in word),
        'has_punct': any(c in string.punctuation for c in word),
        'has_hyphen': '-' in word,
        'is_lower': word.islower(),
        'is_upper': word.isupper(),
        'word_shape': word_shape(word),
        'postag': postag,
        'stem': stemmer.stem(word),
        'lemma': lemmatizer.lemmatize(word),
        'prefix1': word[:1],
        'prefix2': word[:2],
        'suffix1': word[-1:],
        'suffix2': word[-2:]
    }

@st.cache_data
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

@st.cache_data
def sent2labels(sent):
    return [label for token, postag, label in sent]

@st.cache_data
def sent2tokens(sent):
    return [token for token, postag, label in sent]
