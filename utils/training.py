import joblib
import sklearn_crfsuite

from utils.data_loader import load_conll_data
from utils.features import sent2features, sent2labels

train_sents = load_conll_data("data/conll2003_train.txt")
train_sents_fin = load_conll_data("data/conll_sec_data_train.txt")

train_sents.extend(train_sents_fin)
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)
crf.fit(X_train, y_train)

joblib.dump(crf, "models/crf_model.pkl")