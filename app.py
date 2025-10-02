import streamlit as st
import nltk

# Setup NLTK
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define the pages
main_page = st.Page('pages/main_page.py', title='🔎 Part-of-Speech Tagging and Named Entity Recognition')
pos_tag_page = st.Page('pages/pos_tagging.py', title='🏷️ POS Tagging')
ner_crf_page = st.Page('pages/ner_crf_model.py', title='🤖 NER and CRF Model')
corpus_exploration_page = st.Page('pages/corpus.py', title='🧭 Corpus Exploration')
live_demo_page = st.Page('pages/live_demo.py', title='📝 Live Demo')

# Set up Navigation
pg = st.navigation([
    main_page,
    pos_tag_page,
    ner_crf_page,
    corpus_exploration_page,
    live_demo_page
])

pg.run()

# --- Footer ---
st.markdown("---")
st.markdown("""
    Built by Dushyant Goel • [Github](https://github.com/dushyant-goel) • [LinkedIn](https://www.linkedin.com/in/dusdusdushyant-goel-fintech/)   
    🎓 MSc Data Science (Financial Technology), University of Bristol 
""")