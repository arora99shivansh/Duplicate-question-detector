from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from fuzzywuzzy import fuzz

app = Flask(__name__)

# load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

EXPECTED_FEATURES = model.n_features_in_   # 6022

# ---------------- preprocessing ----------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- feature generator ----------------
def get_features(q1, q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # BOW
    bow = cv.transform([q1, q2]).toarray()
    features = np.hstack((bow[0], bow[1]))

    # fuzzy features
    fuzzy = np.array([
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ])

    features = np.hstack((features, fuzzy))

    # -------- FORCE FEATURE SIZE MATCH --------
    if features.shape[0] < EXPECTED_FEATURES:
        pad = EXPECTED_FEATURES - features.shape[0]
        features = np.hstack((features, np.zeros(pad)))
    elif features.shape[0] > EXPECTED_FEATURES:
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

    # 🔥 HARD RULE: exact same question
    if q1.strip().lower() == q2.strip().lower():
        return render_template(
            'index.html',
            prediction="Duplicate Questions ✅"
        )

    X = get_features(q1, q2)
    pred = model.predict(X)[0]

    result = "Duplicate Questions ✅" if pred == 1 else "Not Duplicate ❌"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
