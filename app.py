from flask import Flask, request, jsonify, render_template
import os
import cohere
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

HERE = os.path.dirname(__file__)
app = Flask(__name__)

# Initialize Cohere client (support legacy OPENAI_API_KEY env var for convenience)
cohere_api_key = os.environ.get('COHERE_API_KEY') or os.environ.get('OPENAI_API_KEY')
if not cohere_api_key:
    raise ValueError(
        "COHERE_API_KEY environment variable is not set. "
        "Get your API key at https://cohere.ai/ and set it as an environment variable named COHERE_API_KEY."
    )

client = cohere.Client(api_key=cohere_api_key)

# Configure Cohere model (default to a command-like model for instruction-following)
COHERE_MODEL = os.environ.get('COHERE_MODEL', os.environ.get('OPENAI_MODEL', 'command-xlarge-nightly'))
# Configure analysis temperature (default to 0.3 for more consistent, factual responses)
COHERE_TEMPERATURE = float(os.environ.get('COHERE_TEMPERATURE', os.environ.get('OPENAI_TEMPERATURE', '0.3')))

# --- Diagnostic logging ---
import logging
logger = logging.getLogger('fake-news-detector')
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

logger.info("Cohere API integration initialized with model: %s", COHERE_MODEL)

def analyze_news_with_cohere(text):
    """
    Analyze news text using Cohere to determine if it's fake or real.
    Also performs fact-checking to verify claims.
    """
    try:
        # System prompt that instructs the AI to act as a fact-checker
        system_prompt = """You are an expert fact-checker and news analyst. Your job is to:
1. Analyze the given news text for credibility
2. Identify specific claims that can be fact-checked
3. Assess the likelihoods of the news being real or fake based on:
   - Language patterns (sensationalism, emotional manipulation, clickbait)
   - Logical consistency
   - Presence of verifiable facts vs unsubstantiated claims
   - Common fake news characteristics

Respond in JSON format with the following structure:
{
    "prediction": "Real" or "Fake",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your assessment",
    "claims_identified": ["list of key claims found in the text"],
    "red_flags": ["list of suspicious elements if any"],
    "fact_check_summary": "Summary of fact-checking analysis"
}"""

        # Call Cohere Chat API (Generate was removed Sept 15 2025)
        # Note: Cohere's chat() uses 'message' (singular) not 'messages'
        try:
            # Combine system prompt and user message into a single prompt
            full_message = f"{system_prompt}\n\nAnalyze this news text:\n\n{text}"
            
            response = client.chat(
                model=COHERE_MODEL,
                message=full_message,
                max_tokens=512,
                temperature=COHERE_TEMPERATURE
            )
        except AttributeError as e:
            if "'function' object has no attribute" in str(e) or 'create' in str(e):
                logger.error("Cohere client method error: %s", e)
                raise RuntimeError(
                    "Cohere SDK method not found. Please ensure you have upgraded to Cohere SDK v5+. "
                    "Run: pip install --upgrade cohere"
                )
            raise
        except Exception as e:
            msg = str(e)
            if 'Generate API was removed' in msg or 'migrating-from-cogenerate-to-cochat' in msg:
                logger.error("Cohere Generate API removed: %s", msg)
                raise RuntimeError(
                    "Cohere Generate API has been removed. The app now uses the Cohere Chat API. "
                    "See: https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat"
                )
            logger.error("Cohere chat API error: %s", msg)
            raise

        # Extract the text response from the chat call
        # Cohere's chat() returns a response object with a text attribute or message content
        raw = None
        
        # Try to get text from different response shapes
        if hasattr(response, 'text'):
            raw = response.text
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            raw = response.message.content
        elif hasattr(response, 'content'):
            raw = response.content
        else:
            # Fallback: stringify and try to extract JSON
            raw = str(response)

        if not raw:
            logger.error("No text extracted from Cohere response: %s", response)
            raise ValueError("Failed to extract text from Cohere response")

        raw = raw.strip()

        # The model should return a pure JSON object, but guard against extra text
        def _extract_json(s: str):
            s = s.strip()
            if s.startswith('{') and s.endswith('}'):
                return s
            # Try to find a JSON object within text
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                return s[start:end+1]
            return None

        json_text = _extract_json(raw)
        if not json_text:
            logger.error("Cohere returned non-JSON response: %s", raw[:200])
            raise ValueError("Cohere response did not contain a JSON object as expected")

        result = json.loads(json_text)
        logger.info("Cohere analysis completed: %s", result.get('prediction'))
        return result
        
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Cohere response as JSON: %s", e)
        raise ValueError("Invalid response format from Cohere")
    except Exception as e:
        logger.error("Cohere API error: %s", e)
        raise


@app.route('/__diag', methods=['GET'])
def diag():
    """Return simple diagnostic info for deployment debugging."""
    return jsonify({
        'status': 'ok',
        'cohere_model': COHERE_MODEL,
        'api_key_configured': bool(cohere_api_key)
    })


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

        if len(text.strip()) < 5:
            return jsonify({'error': 'Text is too short for analysis (minimum 5 characters)'}), 400

        # Analyze with Cohere
        analysis = analyze_news_with_cohere(text)

        return jsonify({
            'prediction': analysis.get('prediction', 'Unknown'),
            'confidence': f"{float(analysis.get('confidence', 0)):.4f}",
            'reasoning': analysis.get('reasoning', ''),
            'claims_identified': analysis.get('claims_identified', []),
            'red_flags': analysis.get('red_flags', []),
            'fact_check_summary': analysis.get('fact_check_summary', '')
        })
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return jsonify({'error': str(e)}), 500

# For Vercel serverless deployment
app = app

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)