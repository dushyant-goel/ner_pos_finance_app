import string
import nltk
import os
from nltk.stem import PorterStemmer, WordNetLemmatizer


nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

nltk.download('wordnet', download_dir='nltk_data_dir')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def word_shape(word):
    return ''.join(
        'X' if c.isupper() else
        'x' if c.islower() else
        '0' if c.isdigit() else
        '#' for c in word
    )

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

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]
