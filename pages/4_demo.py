import streamlit as st
from utils.data_loader import load_conll_data
from utils.features import sent2features, sent2labels, sent2tokens
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics

st.header("🧠 Conditional Random Fields (CRF) for Named Entity Recognition")

st.subheader(r"""
We follow the approach used by Alvarado (2015) of using  mixed training sets for niche domains.
The model is trained on CoNLL-2003 data which is POS tagged Reuters newswire data. We divide the 
financial documents data from 8 SEC filling into two parts - first 3 documents supplment the training
and the last 5 are used to evaluate and test.           
""")

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
