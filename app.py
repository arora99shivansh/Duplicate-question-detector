from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from fuzzywuzzy import fuzz

app = Flask(__name__)

# ---------------- load artifacts ----------------
def load_artifacts():
    global model, cv, EXPECTED_FEATURES
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('cv.pkl', 'rb') as f:
        cv = pickle.load(f)
    EXPECTED_FEATURES = model.n_features_in_

load_artifacts()

# ---------------- preprocessing ----------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- simple semantic rule ----------------
def simple_semantic_match(q1, q2):
    q1_tokens = set(preprocess(q1).split())
    q2_tokens = set(preprocess(q2).split())

    if not q1_tokens or not q2_tokens:
        return False

    common_ratio = len(q1_tokens & q2_tokens) / min(len(q1_tokens), len(q2_tokens))

    return common_ratio >= 0.6   # 🔥 threshold (60%)

# ---------------- feature generator ----------------
def get_features(q1, q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    bow = cv.transform([q1, q2]).toarray()
    features = np.hstack((bow[0], bow[1]))

    fuzzy = np.array([
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ])

    features = np.hstack((features, fuzzy))

    if features.shape[0] < EXPECTED_FEATURES:
        features = np.hstack(
            (features, np.zeros(EXPECTED_FEATURES - features.shape[0]))
        )
    else:
        features = features[:EXPECTED_FEATURES]

    return features.reshape(1, -1)

# ---------------- routes ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    q1 = request.form['q1']
    q2 = request.form['q2']

    # 1️⃣ Exact match (bulletproof)
    if preprocess(q1) == preprocess(q2):
        return render_template(
            'index.html',
            prediction="Duplicate Questions ✅"
        )

    # 2️⃣ Simple semantic rule (FIXES YOUR ISSUE)
    if simple_semantic_match(q1, q2):
        return render_template(
            'index.html',
            prediction="Duplicate Questions ✅"
        )

    # 3️⃣ ML model
    X = get_features(q1, q2)
    pred = model.predict(X)[0]

    result = "Duplicate Questions ✅" if pred == 1 else "Not Duplicate ❌"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
