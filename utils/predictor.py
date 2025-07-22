import nltk
from nltk import word_tokenize, pos_tag
from utils.features import sent2features
from sklearn_crfsuite import CRF

def tokenize_and_tag(text):
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    # Convert to dummy NER-tag format expected: (word, POS, _)
    return [(word, pos, "O") for word, pos in tagged]

def prepare_features(text):
    tagged_sent = tokenize_and_tag(text)
    features = sent2features(tagged_sent)
    tokens = [w for w, p, t in tagged_sent]
    pos_tags = [p for w, p, t in tagged_sent]
    return tokens, pos_tags, features
