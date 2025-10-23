# Fake News Detector — Deployment Notes

This repository contains a small Flask app that serves a scikit-learn logistic regression model and a TF-IDF vectorizer to classify news text as real or fake.

This README covers Option 1: including model pickles in the repo and redeploying to Vercel.

Summary
- If you want the fastest fix: include the `model/` pickles (`logistic_model.pkl` and `tfidf_vectorizer.pkl`) in the repo and redeploy. The app will load them at cold-start.
- A recommended alternative (not covered here) is to host pickles in object storage (S3/R2) and set `MODEL_URL` and `VECTORIZER_URL` env vars in Vercel. See `app.py` for download support.

Steps to include pickles in repo and redeploy (Option 1)

1. Stop ignoring `model/` (if currently ignored)

   If your `.gitignore` includes `model/`, remove or comment that line. Example:

   ```gitignore
# model/
   ```

2. Re-add pickles to git and push

   ```bash
   # Stage the pickles
   git add model/logistic_model.pkl model/tfidf_vectorizer.pkl
   git commit -m "Add trained model pickles for deployment"
   git push origin main
   ```

3. Re-deploy on Vercel

   - In the Vercel dashboard, re-create the deployment (or trigger a redeploy of the project). The deployment will now include `model/` and the function should load the pickles on cold-start.

4. Smoke test (after deploy)

   From your machine (replace `https://your-vercel-app.vercel.app` with your deployment URL):

   ```bash
   curl -X POST https://your-vercel-app.vercel.app/predict \
     -H 'Content-Type: application/json' \
     -d '{"text": "This is a test news article to classify."}' -v
   ```

Notes and warnings
- Storing model pickles in the repo is fine for quick fixes but may add size to the repo and is not ideal for production. Consider Option 2 (object storage + env vars) for long-term.
- Make sure `requirements.txt` pins `numpy` and `scipy` versions (already pinned) to prevent binary incompatibilities when loading pickles.

If you want, I can push the README and a tiny smoke-test script now and walk you through re-adding the pickles.

Explicit safe commands to re-track `model/` if it was previously ignored

```bash
# If model/ is listed in .gitignore, temporarily remove/comment that line.
# Then:
git add -f model/logistic_model.pkl model/tfidf_vectorizer.pkl
git commit -m "Add trained model pickles for deployment"
git push origin main

# If you used -f to force-add because of .gitignore, update .gitignore afterward to remove the 'model/' entry
```

What to expect after pushing

- Vercel will run a build and include the `model/` directory in the deployment bundle. The function will attempt to open `model/tfidf_vectorizer.pkl` and `model/logistic_model.pkl` at cold-start and should succeed if the files are there and the Python environment matches the one used to create the pickles.

# fake-news-detector
after a while
