import streamlit as st

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
| Treebank Name                     | Description                                                                                           |
|---------------------              |-------------                                                                                          |
| **Penn Treebank (PTB)**           | The most widely used treebank in English NLP. Introduced in the 1990s. Uses ~50,000 sentence from WSJ |
| **Universal Dependencies (UD)**   | A modern, multilingual POS + dependency grammar project.                                              |
| **Brown Corpus**                  | One of the earliest corpora, used in NLTK. Simpler tagset.                                            |
| **OntoNotes**                     | Used in CoNLL shared tasks. Rich with POS, coreference, and semantics.                                |
""")

st.image("assets/penn_treebank.png", caption="A sentence from the Penn Treebank with POS and phrase structure annotations")

"---"

st.subheader("🔧 How are POS Taggers Models Trained?")

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
""")

"---"

st.subheader(""" 🧰 Libraries Commonly Used """)

st.markdown(r"""

Several NLP libraries provide ready-to-use POS taggers:

| Library    | Description |
|------------|-------------|
| **NLTK**   | Lightweight, educational. Used in this project for pre-tagging. |
| **spaCy**  | Fast, production-ready. Built on neural nets |
| **Stanza** | Deep learning tagger from Stanford NLP. |
| **Flair**  | Contextual embeddings + sequence tagging from Zalando. |

""")