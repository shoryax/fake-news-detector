# 📰 Fake News Detector (Cohere Edition)

An **AI-powered Flask web app** that classifies news articles as *real* or *fake* using **Cohere's AI models** with advanced fact-checking capabilities.
Built with **Python, Cohere API, and Flask**, and designed for deployment on **Vercel** for scalability and low-latency inference.

---

## 📚 Overview

This project demonstrates an **AI-powered fact-checking and news analysis pipeline**:

* Analyze news text using **Cohere’s language models** (default: `command-r-plus` or your preferred model)
* Perform **intelligent fact-checking** to identify claims and verify credibility
* Assess articles based on language patterns, logical consistency, and fake news characteristics
* Serve predictions through a **Flask API**

This application uses state-of-the-art language models for comprehensive news analysis and is intended for educational and demonstration purposes.

---

## ⚙️ How It Works

1. **User sends text** to the `/predict` endpoint.
2. **Cohere API** analyzes the text using a specialized fact-checking prompt.
3. The **AI model** evaluates credibility based on:

   * Language patterns (sensationalism, emotional manipulation)
   * Logical consistency
   * Verifiable facts vs unsubstantiated claims
   * Common fake news characteristics
4. Flask returns a detailed JSON response with prediction, confidence, reasoning, identified claims, and fact-check summary.

---

## 🚀 Deployment

This application is deployed on **Vercel**, running on **Azure Virtual Machine infrastructure**.

### Required Environment Variables

You must set the following environment variable in your Vercel project settings:

* `COHERE_API_KEY` — Your Cohere API key (get one at [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys))

Optional environment variables:

* `COHERE_MODEL` — Model to use (default: `command-r-plus`)

---

### Setting Environment Variables in Vercel

1. Go to your Vercel project dashboard
2. Navigate to **Settings → Environment Variables**
3. Add:

| Key              | Value                 |
| ---------------- | --------------------- |
| `COHERE_API_KEY` | Your Cohere API key   |
| `COHERE_MODEL`   | (optional) model name |

4. Redeploy your application

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

1. **Get a Cohere API key** from [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)
2. **Clone this repository**
3. **Set environment variables**

---

### Local Setup

1. Clone the repository:

```bash
git clone https://github.com/shoryax/fake-news-detector.git
cd fake-news-detector
```

2. Create a `.env` file (use `.env.example` as template):

```bash
cp .env.example .env
# Edit .env and add your COHERE_API_KEY
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Flask app:

```bash
flask run
# or
python app.py
```

5. Test the endpoint:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sample headline for prediction."}'
```

---

## 🚀 Vercel Deployment

1. Fork or clone this repository
2. Connect your repository to **Vercel**
3. Add environment variables:

| Key              | Value               |
| ---------------- | ------------------- |
| `COHERE_API_KEY` | Your Cohere API key |

4. Deploy!

---

## 🔍 Smoke Test (after deployment)

```bash
curl -X POST https://your-vercel-app.vercel.app/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is a test news article to classify."}' -v
```

Or use the provided smoke test script:

```bash
python scripts/smoke_test.py https://your-vercel-app.vercel.app
```

---

## ⚠️ Notes & Best Practices

* **API Key Security** — Never commit `COHERE_API_KEY` to your repository
* **Cost Management** — Cohere billing dashboard: [https://dashboard.cohere.com](https://dashboard.cohere.com)
* **Rate Limiting** — Consider rate limiting for production
* **Model Selection** — Change the model using `COHERE_MODEL` (e.g., `command-r-plus`, `command-light`, etc.)

---

## 🧠 Tech Stack

| Component          | Description                           |
| ------------------ | ------------------------------------- |
| **Python**         | Core programming language             |
| **Flask**          | Web framework for serving predictions |
| **Cohere AI**      | AI-powered fact-checking and analysis |
| **command-r-plus** | Default Cohere model                  |
| **Vercel**         | Cloud platform for deployment         |
| **Azure VM**       | Underlying compute for runtime        |

---

## 🗂️ Repository Structure

```
fake-news-detector/
│
├── app.py                    # Flask app with Cohere integration
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
