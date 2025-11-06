from flask import Flask, request, jsonify, render_template
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

HERE = os.path.dirname(__file__)
app = Flask(__name__)

# Initialize OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=openai_api_key)

# Configure OpenAI model (default to gpt-4o-mini for cost efficiency)
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')

# --- Diagnostic logging ---
import logging
logger = logging.getLogger('fake-news-detector')
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

logger.info("OpenAI API integration initialized with model: %s", OPENAI_MODEL)

def analyze_news_with_openai(text):
    """
    Analyze news text using OpenAI API to determine if it's fake or real.
    Also performs fact-checking to verify claims.
    """
    try:
        # System prompt that instructs the AI to act as a fact-checker
        system_prompt = """You are an expert fact-checker and news analyst. Your job is to:
1. Analyze the given news text for credibility
2. Identify specific claims that can be fact-checked
3. Assess the likelihood of the news being real or fake based on:
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

        # Call OpenAI API
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this news text:\n\n{text}"}
            ],
            temperature=0.3,  # Lower temperature for more consistent, factual responses
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        logger.info("OpenAI analysis completed: %s", result.get('prediction'))
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error("Failed to parse OpenAI response as JSON: %s", e)
        raise ValueError("Invalid response format from OpenAI")
    except Exception as e:
        logger.error("OpenAI API error: %s", e)
        raise


@app.route('/__diag', methods=['GET'])
def diag():
    """Return simple diagnostic info for deployment debugging."""
    return jsonify({
        'status': 'ok',
        'openai_model': OPENAI_MODEL,
        'api_key_configured': bool(openai_api_key)
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

        if len(text.strip()) < 10:
            return jsonify({'error': 'Text is too short for analysis (minimum 10 characters)'}), 400

        # Analyze with OpenAI
        analysis = analyze_news_with_openai(text)
        
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