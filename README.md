# 🧠 Named Entity Recognition (NER) and POS Tagging: A Streamlit Tutorial App

This interactive Streamlit app is both a **tutorial** and a **live demo** for understanding and experimenting with **Part-of-Speech (POS) Tagging** 
and **Named Entity Recognition (NER)** using **Conditional Random Fields (CRFs)**.

## 🧪 Demo

▶️ [App Demo](https://ner-pos-finance-app.streamlit.app/)

---

## 🚀 Features

✅ **Step-by-step tutorial layout** — Theory first, followed by interactive demo  
✅ **POS and NER feature engineering walkthrough**  
✅ **Train and evaluate your own CRF model**  
✅ **Try live entity prediction on custom text**

---

## 🧱 App Structure

The app is organized into pages for clarity:

1. **📖 Theory** – Understand what POS tagging and NER are, why they matter, and how CRFs help.
2. **📖 Part of Speech Tagging** - Explore POS Trees and demystify the trained libraries available like NLTK.
3. **📖 Named Entity Recognition** - Uncover the intution and maths behind **Conditional Random Fields** and how it compares
  to other NER training methods.
4. **Corpus Exploration** - Explanation of data used.
5. **📝 Live Demo** – Enter a sentence and get back predicted POS tags and named entities. Including:
  **🛠️ Feature Engineering & Training** – Dive into how features are extracted for CRF, and train your own model.
  **📈 Model Evaluation** – View precision, recall, and F1-scores by entity type.


---

## 📦 Technologies Used and References

- `Python 3.10`
- [`Streamlit`]
- `nltk`, `sklearn-crfsuite`, `pandas`, `numpy`
- `CoNLL-2003` dataset (with financial SEC extension based on [1] https://aclanthology.org/U15-1010.pdf


[1]J. Cesar, S. Alvarado, K. Verspoor, and T. Baldwin, “Domain Adaption of Named Entity Recognition to Support Credit Risk Assessment”, 
Accessed: Jul. 22, 2025. [Online]. Available: http://www.sec.gov/Archives/edgar/data/1593034/                                                                            

---

## 💻 Run Locally

```bash
git clone https://github.com/yourusername/ner-pos-tutorial.git
cd ner-pos-tutorial
pip install -r requirements.txt
streamlit run Home.py
