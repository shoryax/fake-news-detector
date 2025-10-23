from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# Download NLTK if needed (quiet)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

MODEL_PATH = 'model/logistic_model.pkl'
VECTORIZER_PATH = 'model/tfidf_vectorizer.pkl'

# lazy-loaded model/vectorizer (helps serverless deployments)
model = None
vectorizer = None
_model_lock = None

stop_words = set(stopwords.words('english'))


def _ensure_model_loaded():
    """Load the vectorizer and model on first use."""
    global model, vectorizer, _model_lock
    if model is not None and vectorizer is not None:
        return
    import threading
    if _model_lock is None:
        _model_lock = threading.Lock()
    with _model_lock:
        if model is None or vectorizer is None:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
                raise FileNotFoundError("Run preprocess.py first to generate models!")
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)

def preprocess_text(text):
    if not text:
        return ""
    text = str(text).lower()
    # Keep alphanumeric characters and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,\'-]', '', text)
    # Remove stopwords but keep numbers
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Ensure model/vectorizer are loaded (lazy for serverless)
        _ensure_model_loaded()
        processed = preprocess_text(text)
        features = vectorizer.transform([processed])
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][prediction]

        result = 'Fake' if prediction == 0 else 'Real'
        return jsonify({
            'prediction': result,
            'confidence': f"{float(confidence):.4f}",
            'processed_text': processed[:200] + '...' if len(processed) > 200 else processed  # Optional preview
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Vercel serverless deployment
app = app

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)