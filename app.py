from flask import Flask, request, jsonify, render_template
import pickle
import re
import os
import tempfile
import urllib.request
import urllib.error
import threading
import nltk
from nltk.corpus import stopwords

# If this repo bundles NLTK data (nltk_data/), make sure NLTK finds it.
HERE = os.path.dirname(__file__)
NLTK_DATA_DIR = os.path.join(HERE, 'nltk_data')
if os.path.isdir(NLTK_DATA_DIR):
    import nltk.data
    nltk.data.path.append(NLTK_DATA_DIR)

app = Flask(__name__)

# Paths inside the deployment (relative to project root)
MODEL_PATH = 'https://blissstorage12345.blob.core.windows.net/pickles/logistic_model.pkl?se=2025-11-22T19%3A49Z&sp=r&sv=2022-11-02&sr=b&sig=0vZXLvsH5%2BalFYwLcGZHZQ6ZBenuld9wnjL%2B9VXhUoY%3D'
VECTORIZER_PATH = 'https://blissstorage12345.blob.core.windows.net/pickles/tfidf_vectorizer.pkl?se=2025-11-22T19%3A49Z&sp=r&sv=2022-11-02&sr=b&sig=Khb%2FPuU6Kxl%2BDgILQg89QPxJP6%2F1a2QOL%2BFJBPmy2XQ%3D'

# lazy-loaded model/vectorizer (helps serverless deployments)
model = None
vectorizer = None
_model_lock = threading.Lock()

# stop_words is initialized lazily in _ensure_resources()
stop_words = None

# --- Diagnostic logging ---
import logging
logger = logging.getLogger('fake-news-detector')
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

def _log_deployment_state():
    try:
        vec_exists = os.path.exists(VECTORIZER_PATH)
        model_exists = os.path.exists(MODEL_PATH)
        vec_size = os.path.getsize(VECTORIZER_PATH) if vec_exists else None
        model_size = os.path.getsize(MODEL_PATH) if model_exists else None
    except Exception:
        vec_exists = model_exists = False
        vec_size = model_size = None
    env_vec = bool(os.environ.get('VECTORIZER_URL') or os.environ.get('VEC_URL') or os.environ.get('VECTORIZER'))
    env_model = bool(os.environ.get('MODEL_URL') or os.environ.get('MODEL'))
    logger.info(f"Vercel diagnostic: vectorizer_exists={vec_exists} size={vec_size} model_exists={model_exists} size={model_size} env_vec={env_vec} env_model={env_model}")

# Log an initial deployment state at import time (Vercel will show this in build/runtime logs)
_log_deployment_state()


def _ensure_model_loaded():
    """Load the vectorizer and model on first use."""
    global model, vectorizer, _model_lock
    if model is not None and vectorizer is not None:
        return

    # Ensure NLTK resources and make best-effort to fetch missing model files
    _ensure_resources()

    with _model_lock:
        if model is None or vectorizer is None:
            # If local pickles are missing, try to download from URLs provided via env vars.
            if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
                vec_url = os.environ.get('VECTORIZER_URL') or os.environ.get('VEC_URL') or os.environ.get('VECTORIZER')
                model_url = os.environ.get('MODEL_URL') or os.environ.get('MODEL')
                if vec_url and model_url:
                    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
                    try:
                        _download_file(vec_url, VECTORIZER_PATH)
                        _download_file(model_url, MODEL_PATH)
                    except Exception as e:
                        raise RuntimeError(f"Failed to download model artifacts: {e}")
                else:
                    raise FileNotFoundError("Model files not found. Provide MODEL_URL and VECTORIZER_URL environment variables or include model pickles in the deployment.")

            # Load the pickles
            try:
                with open(VECTORIZER_PATH, 'rb') as f:
                    vectorizer = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load vectorizer pickle: {e}")
            try:
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load model pickle: {e}")


def _download_file(url, dest_path, timeout=20):
    """Download a file from `url` to `dest_path` (atomic write).

    Uses urllib.request (safe in restricted environments). Raises on non-200.
    """
    # Write to a temporary file first
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir)
    os.close(fd)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib/3'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Download failed with status {resp.status} for {url}")
            data = resp.read()
        with open(tmp_path, 'wb') as out:
            out.write(data)
        os.replace(tmp_path, dest_path)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error downloading {url}: {e}")
    except Exception:
        # Clean up temp file on any error
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _ensure_resources():
    """Ensure NLTK resources (stopwords) are available and set `stop_words`.

    This is lazy and resilient: if stopwords cannot be loaded, we fall back to a small built-in list.
    """
    global stop_words
    if stop_words:
        return
    try:
        # Attempt to load bundled or system stopwords
        stop_words = set(stopwords.words('english'))
        return
    except LookupError:
        # Try to download quietly (might fail in locked-down envs)
        try:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
            return
        except Exception:
            # Fallback small list (keeps preprocessing running)
            stop_words = set(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'an', 'for', 'on', 'with', 'as', 'by', 'at', 'from', 'that', 'this', 'it'])


    @app.route('/__diag', methods=['GET'])
    def diag():
        """Return simple diagnostic info useful for Vercel logs.

        - Whether model files exist and sizes
        - Whether MODEL_URL/VECTORIZER_URL are provided
        """
        try:
            vec_exists = os.path.exists(VECTORIZER_PATH)
            model_exists = os.path.exists(MODEL_PATH)
            vec_size = os.path.getsize(VECTORIZER_PATH) if vec_exists else None
            model_size = os.path.getsize(MODEL_PATH) if model_exists else None
        except Exception as e:
            return jsonify({'error': f'Filesystem error: {e}'}), 500
        env_vec = bool(os.environ.get('VECTORIZER_URL') or os.environ.get('VEC_URL') or os.environ.get('VECTORIZER'))
        env_model = bool(os.environ.get('MODEL_URL') or os.environ.get('MODEL'))
        return jsonify({
            'vectorizer_exists': vec_exists,
            'vectorizer_size': vec_size,
            'model_exists': model_exists,
            'model_size': model_size,
            'env_var_vectorizer_present': env_vec,
            'env_var_model_present': env_model
        })

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