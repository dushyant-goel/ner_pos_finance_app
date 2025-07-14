import streamlit as st

st.set_page_config(page_title="NER & POS Tagging")
st.title("🔎 Named Entity Recognition & Part-of-Speech Tagging in Financial Text")
st.markdown(r"""
Welcome! This app demonstrates **Named Entity Recognition (NER)** and **Part-of-Speech (POS)** tagging
using financial news. You can interact with the text, explore annotations, and learn how these NLP tools work.""")


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

st.subheader("📌 Our Dataset")

st.markdown(r"""
For this project, we use a dataset derived from **Alvarado (2015)** — a set of financial 
agreements that were pre-tokenized and POS-tagged using the **NLTK** library.

Under the hood, NLTK uses a tagger trained on the **Brown Corpus**, one of the earliest 
English treebanks.

Each token in the dataset comes with:
- A word
- Its POS tag (from Brown tagset)
- Its manually labeled NER tag

This POS information is a crucial feature for the **ML model** we use for Named Entity Recognition.
""")

"---"

st.subheader("📑 Dataset: Financial Agreements NER Corpus")
st.markdown(r"""
In this task, we use a further labeled dataset introduced by [**Alvarado (2015)**](https://aclanthology.org/U15-1010.pdf) for Named Entity Recognition in the financial domain.
The dataset consists of **eight financial agreements** filed with the U.S. Securities and Exchange Commission (SEC), totaling approximately **50,000 words**.
Each sentence is annotated in **CoNLL format**, where every line contains:

```<token> <POS tag> <NER label>```
            
The dataset includes four named entity labels:
- `ORG`: Organization names (e.g., Barclays Capital)
- `LOC`: Locations (e.g., New York, California)
- `PER`: Person names (e.g., John Doe)
- `MISC`: Miscellaneous entities

An empty line separates each sentence.
""")

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