import streamlit as st

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

