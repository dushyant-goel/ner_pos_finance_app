import streamlit as st

st.set_page_config(page_title="POS Tagging & NER")
st.title("🔎 Part-of-Speech Tagging and Named Entity Recognition")
st.markdown(r"""
Welcome! This app demonstrates **Named Entity Recognition (NER)** and **Part-of-Speech (POS)** tagging.
You will learn how these NLP tools work with theory and interactive tools.""")


st.subheader(r"""
📌 Motivation: Why NER and POS tagging matter in Finance

In the financial sector, **documents like contracts, regulatory filings, and earnings reports** 
are dense with important information—names of parties, legal obligations, amounts, dates, and conditions. 
Extracting this information is critical for:

    📉 Risk assessment (e.g., liabilities, obligations, exposure)
    🧾 Contract summarization (e.g., who owes what to whom)
    🔍 Compliance checks (e.g., detecting location-based restrictions)
    💹 Downstream analytics (e.g., linking entities to market behavior)
    👀 Sentiment analysis (e.g., towards particular product or company)


To perform such tasks, we must first identify and extract entities from the document.
To do this, we can either employ an army of interns or graduates or train ML models to do it.
More and more it's the latter - AI - being employed to do such tasks, with a human verifying
mission critical results. Step through this tutorial to learn the concepts behind POS tagging
and Named Entity Recognition. Try out the interactive features to tune and try out different models
and finally, upload your own documents to recognize the entities in it.

#### ✍️ POS Tagging as a Foundation for Entity Extraction.

Part-of-Speech (POS) tagging assigns grammatical labels to sentences in a document:

    Nouns (NN, NNP): Names of people, companies, instruments
    Verbs (VB): Actions like "acquire", "transfer", "guarantee"
    Adjectives (JJ): Descriptors that modify nouns (e.g., “secured”, “floating”)

This is a crucial first step because:

    Proper nouns (NNP) are often names of people or organizations
    Noun phrases like "collateral agreement" or "parties to agreement" carry semantic weight
    Verbs signal events (e.g., “terminated”, “entered into”, “assigned”)

""")

st.header("🧠 Part-of-Speech (POS) Tagging")

st.markdown(r"""
Remember those **syntax trees** exercise from English lectures in middle school? We drew lines  
sentence to break it down into constituent parts. First we would break the sentence into subject 
(noun phrase) and predicate (verb phrase), and then iteratively break these into determiners, adjective, 
adverbs, etc. The end results looked like an upside down tree, with words at the leaves, each labeled 
with a part of speech.
            
Here's a short explainer video to jog your memory:
""")

st.video("https://youtu.be/CeuhQ3s-Jss?si=uTSqnLbuLStY96En&t=188", start_time="188s")

st.markdown(r"""
This exercise is the foundation of POS tagging. Various universties painstakingly created
and made available massive human hand-annotated (or atleast, human-verified) dataset called 
Treebanks. 
""")

st.markdown(f"#### 🧾 What is a Treebank?")

st.markdown(r"""
A **treebank** is a manually annotated database where each word in a sentence is labeled with:
- Its **Part-of-Speech (POS)** tag
- Its **syntactic role** in the sentence — forming a tree structure

Treebanks are used to **train and evaluate** NLP models for **supervised POS tagging**.
Some well-known gold standard treebank are:
""")

st.markdown(r"""
| Treebank Name       | Description |
|---------------------|-------------|
| **Penn Treebank (PTB)** | The most widely used treebank in English NLP. Introduced in the 1990s. Uses ~50,000 sentence from WSJ |
| **Universal Dependencies (UD)** | A modern, multilingual POS + dependency grammar project. |
| **Brown Corpus**    | One of the earliest corpora, used in NLTK. Simpler tagset. |
| **OntoNotes**       | Used in CoNLL shared tasks. Rich with POS, coreference, and semantics. |
""")

st.image("assets/penn_treebank.png", caption="A sentence from the Penn Treebank with POS and phrase structure annotations")


st.subheader("🔧 How are POS Taggers Built?")

st.markdown(r"""
POS taggers are trained on these treebanks using various methods:

- 🎲 **Probabilistic Models** (like Hidden Markov Models - HMMs)
- 🧮 **Generative Models** (e.g., Maximum Entropy Markov Models)
- 🔀 **Recurrent Neural Networks (RNNs) / LSTM**
- ⚙️ **Modern Transformers / BERT-style models**

Each model learns to **predict the most likely tag** for a word based on its context and history.

These models are often evaluated on the same treebanks they were trained on, with metrics like 
**Precision, Recall, F1 Score**, and **Accuracy**. Here are the benchmarks from the Penn Treebank
""")

st.image("assets/penn_pos_benchmark.png", caption="Penn Treebank POS model benchmarks (accessed: 14 Jul 2025)")

st.markdown(r"""
Models routinely achieve 97% accuracy for English corpus. These pretrained models are made
also freely and publically available along with training parameter details to download with
NLP Libraries such as NLTK and spaCy.
            
#### 🧰 Libraries Commonly Used

Several NLP libraries provide ready-to-use POS taggers:

| Library    | Description |
|------------|-------------|
| **NLTK**   | Lightweight, educational. Used in this project for pre-tagging. |
| **spaCy**  | Fast, production-ready. Built on neural nets |
| **Stanza** | Deep learning tagger from Stanford NLP. |
| **Flair**  | Contextual embeddings + sequence tagging from Zalando. |

""")

st.subheader("📌 Dataset: Financial Agreements NER Corpus")

st.markdown(r"""
In this task, we use a further labeled dataset introduced by [**Alvarado (2015)**](https://aclanthology.org/U15-1010.pdf) 
for Named Entity Recognition in the financial domain. The dataset consists of **eight financial agreements** 
filed with the U.S. Securities and Exchange Commission (SEC), totaling approximately **50,000 words**.
that were pre-tokenized and POS-tagged using **NLTK**. Under the hood, NLTK uses a tagger trained on the **Brown Corpus**, one of the earliest 
English treebanks.

These <word, POS> pairs were then manually-labelled with a NER tag by the authors. 
The dataset includes four named entity labels:
- `ORG`: Organization names (e.g., Barclays Capital)
- `LOC`: Locations (e.g., New York, California)
- `PER`: Person names (e.g., John Doe)
- `MISC`: Miscellaneous entities

In addition **IO format** is used to tag entities that span multiple tokens. Tokens that occur inside a span are
tagged with an I, and any tokens outside of any span of interest are labeled O.            

After tagging, token in the now dataset comes with:
- A word
- Its POS tag (from Brown tagset)
- Its **manually** labeled NER tag

Each token is annotated in **CoNLL format**, where it contains:
```<token> <POS tag> <NER label>```

An empty line separates each sentence.

This POS/NER information is a serves as the training set for the **ML model** we use for Named Entity Recognition.
""")

"---"

from utils.data_loader import load_conll_data
import pandas as pd

st.subheader("🔍 Sample Labeled Sentences")

conll_sentences = load_conll_data("data/conll_data_train.txt")

# Show just the first 3 sentences
for i, sent in enumerate(conll_sentences[:3]):
    st.markdown(f"**Sentence {i+1}**")
    df = pd.DataFrame(sent, columns=["Token", "POS Tag", "NER Label"])
    st.dataframe(df, use_container_width=True)

st.markdown(r"""
You'll notice:
- Proper nouns (NNP) often map to ORG, PER, or LOC.
- This structure makes the dataset suitable for a **CRF model**, where context and handcrafted features help improve NER performance.
""")

"---"

st.subheader("🔍 Named Entity Recognition with CRFs")
st.markdown(r"""
Extracting named entities (like **Barclays**, **New York**, or **credit support annex**) is a fundamental task in NLP — especially important in **financial documents** where even a single misidentified entity can cause major downstream errors.

One could try to write **manual rules** ("If a word is capitalized, it might be a person or company"), or use **dictionaries** of known names — but this approach breaks down:
- New entities appear constantly (e.g., new companies, people)
- Context matters: "Apple" could be a fruit or a company
- Rules are brittle and hard to maintain

So instead, we use **Conditional Random Fields (CRFs)** — a machine learning model specifically designed to handle **structured sequence prediction**, like tagging each word in a sentence with the right entity label.
""")

st.markdown("""
A **Conditional Random Field** is a type of probabilistic graphical model. It models the **conditional probability** of a sequence of labels (e.g., entity tags) given a sequence of observations (e.g., words and their features).

It’s particularly powerful because:
- It considers the **entire sequence context** (not just one word at a time)
- It can incorporate **handcrafted features** that encode useful heuristics
- It allows you to model **dependencies between tags** (e.g., if the previous tag is B-ORG, the next might be I-ORG)

Mathematically, the CRF assigns a score to each possible sequence of labels and normalizes it into a probability:

""")

st.latex(r"""
P(Y \mid X) = \frac{1}{Z_X} \exp\left( \sum_{i} w_i F_i(X, Y) \right)
""")

st.markdown(r"""
Where:
- $X$ = input sequence (features of words)
- $Y$ = output label sequence (NER tags)
- $F_i(X, Y)$ = feature functions
- $w_i$ = learned weights
- $Z_X$ = normalization constant (partition function)
""")

st.subheader("🧰 What Features Does the CRF Use?")
st.markdown(r"""
To help the model make better decisions, we transform each word into a rich feature set. These features reflect **orthographic, grammatical, and morphological** cues.
Here's the full list used in our model:
""")

st.markdown(r"""
| Feature | Description |
|---------|-------------|
| `word` | The token itself, lowercased |
| `is_capitalized` | Word starts with a capital letter |
| `has_capitals_inside` | Capital letters not at the beginning |
| `has_digits` | Word contains digits |
| `has_punct` | Word contains punctuation |
| `has_hyphen` | Word contains a hyphen |
| `is_upper` | All uppercase |
| `is_lower` | All lowercase |
| `word_shape` | Encodes casing, digits, and symbols (e.g., “Xxxx”, “00-000”) |
| `postag` | Part-of-speech tag (from NLTK) |
| `stem` | Word stem (e.g., “running” → “run”) |
| `lemma` | Word lemma (e.g., “ran” → “run”) |
| `prefix1` / `prefix2` | First 1 or 2 letters |
| `suffix1` / `suffix2` | Last 1 or 2 letters |
""")

st.markdown(r"""
Let's say we have the word **“Lloyds”** in a sentence.

The CRF input might look like:

```json
{
  "word": "lloyds",
  "is_capitalized": true,
  "has_capitals_inside": false,
  "has_digits": false,
  "has_punct": false,
  "has_hyphen": false,
  "is_upper": false,
  "is_lower": true,
  "word_shape": "Xxxxx",
  "postag": "NNP",
  "stem": "lloyd",
  "lemma": "lloyd",
  "prefix1": "l",
  "prefix2": "ll",
  "suffix1": "s",
  "suffix2": "ds"
}
```
Based on patterns across hundreds of such examples, the CRF learns which combinations of 
features predict entity types like ORG or PER.
""")

st.markdown(r"""
- **Naive Bayes / Logistic Regression** treats each word independently
- **HMM** can model sequences but doesn't support custom features well
- **CRF** is the sweet spot — it supports **sequence awareness** *and* **handcrafted features**

That's why it has been a go-to choice for NER tasks before deep learning took over — and still performs very well when trained on domain-specific data.
""")

import streamlit as st
from utils.data_loader import load_conll_data
from utils.features import sent2features, sent2labels, sent2tokens
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics

st.header("🧠 Conditional Random Fields (CRF) for Named Entity Recognition")

# --- Load training and test data ---
train_sents = load_conll_data("data/conll_data_train.txt")
test_sents = load_conll_data("data/conll_data_test.txt")

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

y_pred = crf.predict(X_test)
report = metrics.flat_classification_report(y_test, y_pred, output_dict=True)

df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report.style.format(precision=2))

# --- View features of a test sentence ---
st.subheader("🔍 Explore Features for a Sentence")

sentence_idx = st.slider("Select test sentence index", 0, len(test_sents)-1, 0)
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

import streamlit as st
import joblib
import pandas as pd
from utils.predictor import prepare_features

st.header("🧾 Predict Named Entities in Your Financial Document")

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

    st.dataframe(df)

    # Optional: Highlight entities
    st.subheader("🧠 Extracted Entities")
    for label in set(y_pred):
        if label != "O":
            ents = [t for t, l in zip(tokens, y_pred) if l == label]
            st.markdown(f"**{label}**: {', '.join(ents)}")
