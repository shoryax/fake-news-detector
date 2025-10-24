Sure — here’s a clean, professional, and well-structured **`README.md`** that merges all your deployment notes, adds clarity, and makes it look like a polished open-source project ready for GitHub.

---

# 📰 Fake News Detector

A **machine learning-powered Flask web app** that classifies news articles as *real* or *fake* using a **Logistic Regression** model and **TF-IDF vectorization**.
Built with **Python, scikit-learn, and Flask**, and deployed on **Vercel** (backed by Azure infrastructure) for scalability and low-latency inference.

---

## 📚 Overview

This project demonstrates a practical end-to-end **text classification pipeline**:

* Preprocess text using **TF-IDF vectorization**
* Classify using a **Logistic Regression** model trained on labeled news datasets
* Serve predictions through a **Flask API**

The model was trained on data up to **December 2018** and is intended for educational and demonstration purposes.

---

## ⚙️ How It Works

1. **User sends text** to the `/predict` endpoint.
2. **TF-IDF vectorizer** transforms the text into numerical features.
3. The **trained logistic regression model** predicts whether the text is *real* or *fake*.
4. Flask returns a JSON response containing the classification result.

---

## 🚀 Deployment

This application is deployed on **Vercel**, running on **Azure Virtual Machine infrastructure**.
Deployment includes:

* Flask backend API (`app.py`)
* Model pickles:

  * `model/logistic_model.pkl`
  * `model/tfidf_vectorizer.pkl`

Both files are bundled in the deployment bundle to enable instant loading on cold start.

---

## 🧩 Quick Deployment (Option 1: Include Pickles in Repo)

If you want the **fastest working deployment**, simply include your model pickles in the repository.

### 1. Stop ignoring `model/`

If your `.gitignore` contains this line:

```gitignore
model/
```

Comment it out or remove it:

```gitignore
# model/
```

### 2. Re-add the model files and push

```bash
git add model/logistic_model.pkl model/tfidf_vectorizer.pkl
git commit -m "Add trained model pickles for deployment"
git push origin main
```

### 3. Re-deploy on Vercel

In the **Vercel Dashboard**, trigger a redeployment of your project.
The build will now include the `model/` directory, ensuring Flask can load your pickles at runtime.

---

## 🔍 Smoke Test (after deployment)

Once deployed, run a quick sanity check from your local terminal:

```bash
curl -X POST https://your-vercel-app.vercel.app/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is a test news article to classify."}' -v
```

You should receive a JSON response indicating whether the input text is **real** or **fake**.

---

## ⚠️ Notes & Best Practices

* **Option 1** (bundling pickles) is fine for small projects and quick fixes.
  For production setups, consider **Option 2** (hosting pickles on object storage such as AWS S3 or Cloudflare R2) and set:

  * `MODEL_URL`
  * `VECTORIZER_URL`
    environment variables in your Vercel settings.
* Ensure `requirements.txt` **pins versions** of `numpy`, `scipy`, and `scikit-learn` to avoid binary incompatibilities when unpickling.

Example:

```txt
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.1
Flask==3.0.3
```

---

## 🧪 Local Development

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app locally:

   ```bash
   flask run
   ```

4. Send a test request:

   ```bash
   curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Sample headline for prediction."}'
   ```

---

## 🧠 Tech Stack

| Component               | Description                                   |
| ----------------------- | --------------------------------------------- |
| **Python**              | Core programming language                     |
| **Flask**               | Web framework for serving predictions         |
| **scikit-learn**        | Machine learning library (model + vectorizer) |
| **TF-IDF**              | Feature extraction from raw text              |
| **Logistic Regression** | Classification model                          |
| **Vercel**              | Cloud platform for deployment                 |
| **Azure VM**            | Underlying compute for runtime                |

---

## 🗂️ Repository Structure

```
fake-news-detector/
│
├── app.py                    # Flask application entry point
├── requirements.txt          # Dependencies
├── model/
│   ├── logistic_model.pkl    # Trained logistic regression model
│   └── tfidf_vectorizer.pkl  # TF-IDF vectorizer
├── static/                   # Optional: static assets
├── templates/                # Optional: HTML templates
└── README.md                 # This file
```

---

## 🧾 License

This project is open-sourced under the **MIT License**.
Feel free to use, modify, and distribute with attribution.
