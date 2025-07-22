import streamlit as st

st.subheader("🔍 Named Entity Recognition with CRFs")
st.markdown(r"""
Extracting named entities (like **Barclays**, **New York**, or **credit support annex**) is a fundamental task in NLP — especially important in **financial documents** where even a single misidentified entity can cause major downstream errors.

One could try to write **manual rules** ("If a word is capitalized, it might be a person or company"), or use **dictionaries** of known names — but this approach breaks down:
- New entities appear constantly (e.g., new companies, people)
- Context matters: "Apple" could be a fruit or a company
- Rules are brittle and hard to maintain

So instead, we use **Conditional Random Fields (CRFs)** — a machine learning model specifically designed to handle **structured sequence prediction**, like tagging each word in a sentence with the right entity label.
""")

st.markdown(r"""
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