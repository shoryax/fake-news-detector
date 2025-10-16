import pandas as pd
import numpy as np
import csv
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import os

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)

# Quick file inspection (add this temporarily before loading)
print("=== Inspecting raw train.csv ===")
with open('data/train.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 5:  # First 5 lines
            print(f"Line {i}: {repr(line[:300])}...")  # repr shows quotes/escapes
        if i >= 5:
            break

# Step 1: Load and Explore Dataset
print("=== Step 1: Loading and Exploring Dataset ===")
def load_csv_safe(file_path):
    df = pd.read_csv(
        file_path,
        sep=';',  # Key fix: Semicolon delimiter
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        doublequote=True,
        escapechar='\\',
        on_bad_lines='warn',
        engine='python',
        skipinitialspace=True
    )
    # Drop unnamed index if present (common artifact)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

train_df = load_csv_safe('data/train.csv')
val_df = load_csv_safe('data/validation.csv')
test_df = load_csv_safe('data/test.csv')

# Exploration
print("Train Dataset Info:")
print(f"Columns: {train_df.columns.tolist()}")
print(train_df.head())
print(f"Shape: {train_df.shape}")
if 'label' in train_df.columns:
    print(train_df['label'].value_counts())
else:
    print("No 'label' column—check file.")
print(train_df.isnull().sum())
print(f"Loaded rows: {len(train_df)}")

print("\nValidation Shape:", val_df.shape)
print("Test Shape:", test_df.shape)

print("Train Dataset Info:")
print(train_df.head())
print(f"Shape: {train_df.shape}")
print(train_df['label'].value_counts() if 'label' in train_df.columns else "No 'label' column found!")
print(train_df.isnull().sum())

# Clean: Drop missing text
train_df = train_df.dropna(subset=['text'])
val_df = val_df.dropna(subset=['text'])
test_df = test_df.dropna(subset=['text'])
print(f"Cleaned Train Shape: {train_df.shape}")

# Step 2: Text Preprocessing
print("\n=== Step 2: Text Preprocessing ===")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply to train, val, test (use 'text' column; optionally concat title: train_df['full_text'] = train_df['title'].fillna('') + ' ' + train_df['text'])
train_df['processed_text'] = train_df['text'].apply(preprocess_text)
val_df['processed_text'] = val_df['text'].apply(preprocess_text)
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

print("Sample Preprocessed Text:")
print(train_df['processed_text'].head(1).values[0][:200] + "...")  # First 200 chars

# Step 3: Feature Extraction with TF-IDF
print("\n=== Step 3: TF-IDF Vectorization ===")
X_train = train_df['processed_text']
y_train = train_df['label']
X_val = val_df['processed_text']
y_val = val_df['label']
X_test = test_df['processed_text']

# 80/20 split on train for internal train/test (per PDF)
X_train_split, X_temp, y_train_split, y_temp = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# TF-IDF (max_features=5000 for efficiency)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_split)
X_temp_tfidf = vectorizer.transform(X_temp)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Train TF-IDF Shape: {X_train_tfidf.shape}")

# Step 4: Train Logistic Regression Model
print("\n=== Step 4: Training Logistic Regression ===")
param_grid = {'C': [0.1, 1, 10]}
lr = LogisticRegression(random_state=42, max_iter=1000)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_tfidf, y_train_split)

best_model = grid_search.best_estimator_
print(f"Best Params: {grid_search.best_params_}")

# Predict on temp split
y_temp_pred = best_model.predict(X_temp_tfidf)
print(f"Temp Split Accuracy: {accuracy_score(y_temp, y_temp_pred):.4f}")
print(f"Temp Split F1: {f1_score(y_temp, y_temp_pred):.4f}")

# Step 5: Evaluate on Validation and Test, Save Model
print("\n=== Step 5: Evaluation and Saving ===")
# Val eval
y_val_pred = best_model.predict(X_val_tfidf)
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Validation F1: {f1_score(y_val, y_val_pred):.4f}")
print("\nClassification Report (Val):\n", classification_report(y_val, y_val_pred))

# Test predictions (if no labels, just predict; assume labels if present)
if 'label' in test_df.columns:
    y_test = test_df['label']
    y_test_pred = best_model.predict(X_test_tfidf)
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test F1: {f1_score(y_test, y_test_pred):.4f}")
else:
    y_test_pred = best_model.predict(X_test_tfidf)
    print("Test Predictions Generated (no labels for accuracy). Sample:", y_test_pred[:5])

# Save
os.makedirs('model', exist_ok=True)
with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model/logistic_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nModel and vectorizer saved! Ready for Flask backend.")