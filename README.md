# 📰 Fake News Detector

An **AI-powered Flask web app** that classifies news articles as *real* or *fake* using **OpenAI's GPT models** with advanced fact-checking capabilities.
Built with **Python, OpenAI API, and Flask**, and designed for deployment on **Vercel** for scalability and low-latency inference.

---

## 📚 Overview

This project demonstrates an **AI-powered fact-checking and news analysis pipeline**:

* Analyze news text using **OpenAI's GPT models** (default: gpt-4o-mini)
* Perform **intelligent fact-checking** to identify claims and verify credibility
* Assess articles based on language patterns, logical consistency, and fake news characteristics
* Serve predictions through a **Flask API**

This application uses state-of-the-art language models for comprehensive news analysis and is intended for educational and demonstration purposes.

---

## ⚙️ How It Works

1. **User sends text** to the `/predict` endpoint.
2. **OpenAI API** analyzes the text using a specialized fact-checking prompt.
3. The **AI model** evaluates credibility based on:
   - Language patterns (sensationalism, emotional manipulation)
   - Logical consistency
   - Verifiable facts vs unsubstantiated claims
   - Common fake news characteristics
4. Flask returns a detailed JSON response with prediction, confidence, reasoning, identified claims, and fact-check summary.

---

## 🚀 Deployment

This application is deployed on **Vercel**, running on **Azure Virtual Machine infrastructure**.

### Required Environment Variables

You must set the following environment variable in your Vercel project settings:

* `OPENAI_API_KEY` - Your OpenAI API key (get one at https://platform.openai.com/api-keys)

Optional environment variables:

* `OPENAI_MODEL` - Model to use (default: `gpt-4o-mini`)

### Setting Environment Variables in Vercel

1. Go to your Vercel project dashboard
2. Navigate to Settings → Environment Variables
3. Add `OPENAI_API_KEY` with your API key
4. Optionally add `OPENAI_MODEL` if you want to use a different model
5. Redeploy your application

---

## 🔍 API Response Format

The `/predict` endpoint returns a detailed analysis:

```json
{
  "prediction": "Fake" or "Real",
  "confidence": "0.8500",
  "reasoning": "Brief explanation of the assessment",
  "claims_identified": ["list of key claims found"],
  "red_flags": ["suspicious elements if any"],
  "fact_check_summary": "Summary of fact-checking analysis"
}
```

---

## 🧩 Quick Deployment

### Prerequisites

1. **Get an OpenAI API key** from https://platform.openai.com/api-keys
2. **Clone this repository**
3. **Set up environment variables**

### Local Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/shoryax/fake-news-detector.git
   cd fake-news-detector
   ```

2. Create a `.env` file (use `.env.example` as template):

   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app locally:

   ```bash
   flask run
   # or
   python app.py
   ```

5. Send a test request:

   ```bash
   curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Sample headline for prediction."}'
   ```

### Vercel Deployment

1. Fork or clone this repository
2. Connect your repository to Vercel
3. In Vercel project settings, add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
4. Deploy!

---

## 🔍 Smoke Test (after deployment)

Once deployed, run a quick sanity check from your local terminal:

```bash
curl -X POST https://your-vercel-app.vercel.app/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is a test news article to classify."}' -v
```

You should receive a detailed JSON response with prediction, confidence, reasoning, claims identified, and fact-check summary.

You can also use the provided smoke test script:

```bash
python scripts/smoke_test.py https://your-vercel-app.vercel.app
```

---

## ⚠️ Notes & Best Practices

* **API Key Security**: Never commit your `OPENAI_API_KEY` to the repository. Always use environment variables.
* **Cost Management**: The default model `gpt-4o-mini` is cost-efficient. Monitor your OpenAI API usage at https://platform.openai.com/usage
* **Rate Limiting**: Consider implementing rate limiting for production deployments to control costs
* **Model Selection**: You can change the model by setting the `OPENAI_MODEL` environment variable (e.g., `gpt-4o`, `gpt-3.5-turbo`)

---

## 🧠 Tech Stack

| Component    | Description                                |
| ------------ | ------------------------------------------ |
| **Python**   | Core programming language                  |
| **Flask**    | Web framework for serving predictions      |
| **OpenAI**   | AI-powered fact-checking and analysis      |
| **GPT-4o-mini** | Default language model (fast & cost-effective) |
| **Vercel**   | Cloud platform for deployment              |
| **Azure VM** | Underlying compute for runtime             |

---

## 🗂️ Repository Structure

```
fake-news-detector/
│
├── app.py                    # Flask application with OpenAI integration
├── requirements.txt          # Dependencies
├── .env.example              # Example environment variables
├── static/                   # Static assets (CSS, JS)
├── templates/                # HTML templates
├── scripts/
│   └── smoke_test.py         # API testing script
└── README.md                 # This file
```

---

## 🧾 License

This project is open-sourced under the **MIT License**.
Feel free to use, modify, and distribute with attribution.
