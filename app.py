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

# ---------------- intent + topic rule (SHORT QUESTIONS FIX) ----------------
def intent_match(q1, q2):
    q1p = preprocess(q1)
    q2p = preprocess(q2)

    intent_words = {"what", "explain", "define", "meaning", "describe"}

    q1_tokens = set(q1p.split())
    q2_tokens = set(q2p.split())

    # extract core topic (remove intent words)
    q1_topic = q1_tokens - intent_words
    q2_topic = q2_tokens - intent_words

    if not q1_topic or not q2_topic:
        return False

    topic_overlap = len(q1_topic & q2_topic) / min(len(q1_topic), len(q2_topic))

    return topic_overlap >= 0.8

# ---------------- semantic + fuzzy rule ----------------
def semantic_fuzzy_match(q1, q2):
    q1p = preprocess(q1)
    q2p = preprocess(q2)

    q1_tokens = set(q1p.split())
    q2_tokens = set(q2p.split())

    # protect short questions
    if len(q1_tokens) < 4 or len(q2_tokens) < 4:
        return False

    common_ratio = len(q1_tokens & q2_tokens) / min(len(q1_tokens), len(q2_tokens))
    fuzzy_score = fuzz.token_set_ratio(q1p, q2p)

    return common_ratio >= 0.75 and fuzzy_score >= 85

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
    q1 = request.form.get('q1', '')
    q2 = request.form.get('q2', '')

    # 1️⃣ Exact match
    if preprocess(q1) == preprocess(q2):
        return render_template('index.html', prediction="Duplicate Questions ✅")

    # 2️⃣ Intent + topic rule
    if intent_match(q1, q2):
        return render_template('index.html', prediction="Duplicate Questions ✅")

    # 3️⃣ Semantic + fuzzy rule
    if semantic_fuzzy_match(q1, q2):
        return render_template('index.html', prediction="Duplicate Questions ✅")

    # 4️⃣ ML model
    X = get_features(q1, q2)
    pred = model.predict(X)[0]

    result = "Duplicate Questions ✅" if pred == 1 else "Not Duplicate ❌"
    return render_template('index.html', prediction=result)

# ---------------- run app ----------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
