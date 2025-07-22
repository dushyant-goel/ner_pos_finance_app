import streamlit as st

st.subheader("🔍 Named Entity Recognition with CRFs")
st.markdown(r"""
Extracting named entities (like **Barclays**, **New York**, or **credit support annex**) is a fundamental task in NLP — especially important in **financial documents** where even misidentified entity can cause downstream errors.
We want to extract these entities then to track relations between them, to summarize the information about the identified entity or to find the sentiment 
about the entitiy in the documents.

Why not use manual rules to identify entities? Manual identification will ensure that we do not miss out on edge cases and that the identified entities are
indeed correct, as no ML or NLP method can garuntee 100% accuracy.
            
One could try to write **manual rules** ("If a word is capitalized, it might be a person or company"), or use **dictionaries** of known names — but this approach breaks down:
- New entities appear constantly (e.g., new companies, people)
- Context matters: "Apple" could be a fruit or a company
- Rules are brittle and hard to maintain - and diffcult to generalize.

Other automated methods to identify entities (after POS tagging) include **Naive Bayes / Logistic Regression** and **Hidden Markov Models (HMM), however each has their limitations.**
Logistic Regression tags each word independently, missing out context from words around the entity.
HMM does not consider rich features listed below which we use to richly annotate the data to improve extraction.

So instead, we use **Conditional Random Fields (CRFs)** — a machine learning model specifically designed to handle **structured sequence prediction**, 
like tagging each POS identified word in a sentence with the right entity label. We can incorportate arbritary rules - such as alphabet capitalizations but also 
more bespoke rules or sequence of tags after which they appear, word order, character and number sequence, etc.
""")


st.subheader("🧠 CRF Theory")
st.markdown(r"""
A **Conditional Random Field** is a type of probabilistic model. It models the **conditional probability** of a sequence of labels (e.g., entity tags) given a sequence of observations (e.g., words and their features).

It's particularly powerful because:
- It considers the **entire sequence context** (not just one word at a time)
- It can incorporate **handcrafted features** that encode useful heuristics
- It allows you to model **dependencies between tags** (e.g., if the previous tag is B-ORG, the next might be I-ORG)

Mathematically, the CRF assigns a score to each possible sequence of labels and normalizes it into a probability of output
named-entity labels. **POS tag** is one of the main feature for the input.

""")

st.markdown(r"""
Suppose we have $K$ features for every word in a input sequence $X$ (see next section for example of features used). Then we define a function $F$ that maps
each sequence of $X$ to output sequence $Y$, each with weight $w_k$. Then,
         """)

st.latex(r"""
         
  P(Y \mid X) = \frac{ \exp(\sum_{k=1}^{K} w_k F_k(X, Y))}{\sum_{Y' \in Y_{all}}\exp{\sum_{k=1}^{K} w_k F_k(X, Y'})}
""")

st.markdown(r"""
  This gives the probabilty for a labeled sequence Y to be the output for input of pos tagged sequence X. The denominator is 
  extracted as function $Z$ dependend only on the $X$, as sum of $Y \in Y_{all}$ is constant.
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

st.markdown(r"""
The function $F(X, Y)$ is called global feature. We maximize the probability score for each output Y*. 
For calcutating it, we don't brute force our way through the entire search space. We instead use **Viterbi algorithm**
that efficiently prunes the search space for the most probable sequence. We will look deeper into Viterbi algorithm in another app.
""")

st.subheader("🧰 What Features Does the CRF Use?")
st.markdown(r"""
To help the model make better decisions, we transform each word into a rich feature set. These features reflect **orthographic, grammatical, and morphological** cues.
Here's the full list used in our model:
""")

st.markdown(r"""
| Feature               | Description |
|---------              |-------------|
| `word`                | The token itself, lowercased |
| `is_capitalized`      | Word starts with a capital letter |
| `has_capitals_inside` | Capital letters not at the beginning |
| `has_digits`          | Word contains digits |
| `has_punct`           | Word contains punctuation |
| `has_hyphen`          | Word contains a hyphen |
| `is_upper`            | All uppercase |
| `is_lower`            | All lowercase |
| `word_shape`          | Encodes casing, alphabets, digits, and symbols (e.g., “Xxxx”, “00-000”) |
| `postag`              | Part-of-speech tag (from NLTK) |
| `stem`                | Word stem (e.g., “running” → “run”) (stemmer from NLTK) |
| `lemma`               | Word lemma (e.g., “ran” → “run”) (lemmatizer from NLTK) |
| `prefix1`             | Prefix of length 1 |
| `prefix2`             | Prefix of length 2 |
| `suffix1`             | Suffix of length 2 |
| `suffix2`             | Suffix of length 2 |
""")

st.markdown(r"""
Let's say we have the word **Lloyds** in a sentence.

The CRF **input** might look like:

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
- **Naive Bayes / Logistic Regression** treats each word independently.
- **HMM** can model sequences but doesn't support custom features well.
- **CRF** is the sweet spot — it supports **sequence awareness** *and* **handcrafted features**

That's why it has been a go-to choice for NER tasks before deep learning took over — and still performs very well when trained on domain-specific data.
CRF training and inference is also lots less computationally expensive than deep learning tasks. Specifically, if we use it for a nich - such as financial documents
it performs very well.
""")