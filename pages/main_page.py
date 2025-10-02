import streamlit as st

st.set_page_config(page_title="POS Tagging & NER")
st.title("🔎 Part-of-Speech Tagging and Named Entity Recognition")

st.markdown(r"""
Welcome! This app demonstrates **Named Entity Recognition (NER)** and **Part-of-Speech (POS)** tagging.
You will learn how these NLP tools work with theory and interactive tools, taking an example from a corpus of financial agreements.""")

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

#### ✍️ POS Tagging as a Foundation for Named Entity Recognition

Part-of-Speech (POS) tagging assigns grammatical labels to sentences in a document:

    Nouns (NN, NNP): Names of people, companies, instruments
    Verbs (VB): Actions like "acquire", "transfer", "guarantee"
    Adjectives (JJ): Descriptors that modify nouns (e.g., “secured”, “floating”)

This is a crucial first step because:

    Proper nouns (NNP) are often names of people or organizations
    Noun phrases like "collateral agreement" or "parties to agreement" carry semantic weight
    Verbs signal events (e.g., “terminated”, “entered into”, “assigned”)

""")