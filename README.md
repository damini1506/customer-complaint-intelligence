# ✦ ResolveAI — Customer Complaint Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-yellow.svg)](https://www.sbert.net/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ResolveAI** is an end-to-end NLP and Machine Learning platform designed to analyze, classify, prioritize, and route customer support complaints in real time. Combining discriminative classification pipelines with dense semantic embeddings, ResolveAI automates ticket triage, surfaces similar historical resolutions, and accelerates customer support operations.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Machine Learning & NLP Pipeline](#-machine-learning--nlp-pipeline)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [API Reference](#-api-reference)
- [Jupyter Notebooks & Workflow](#-jupyter-notebooks--workflow)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🌟 Overview

Customer service teams frequently face high volumes of incoming tickets that require fast classification, prioritization, and assignment. Manual triage introduces latency and inconsistencies. 

**ResolveAI** resolves this challenge by providing:
1. **Automated Multi-Dimensional Classification**: Categorization, priority scoring, sentiment analysis, and emotion detection from raw complaint text.
2. **Smart Department Routing**: Direct mapping of incoming tickets to responsible departments (e.g., *Billing*, *Technical Support*, *Retention*, *Product*).
3. **Semantic Similarity Search**: Real-time vector similarity matching against historical complaint tickets to identify patterns and past resolutions.
4. **Interactive Web Dashboard**: A modern, responsive user interface with real-time feedback and glassmorphism styling.

---

## 🚀 Key Features

- **Multi-Class Intent & Category Prediction**:
  - Predicts ticket categories such as *Technical issue*, *Billing inquiry*, *Product inquiry*, *Cancellation request*, and *Refund request*.
  - Employs intent-guided regular expression heuristics coupled with trained Machine Learning classifier fallback.

- **Automated Priority Escalation**:
  - Classifies urgency (*Critical*, *High*, *Medium*, *Low*) to prioritize urgent issues and meet SLAs.

- **Sentiment & Emotion Intelligence**:
  - Classifies sentiment into *Positive*, *Neutral*, or *Negative*.
  - Detects granular customer emotion states (*Frustrated*, *Angry*, *Sad*, *Satisfied*, *Not Satisfied*).

- **Intelligent Department Routing**:
  - Directs tickets immediately to the appropriate operational queue:
    - `Technical issue` $\rightarrow$ **Technical Support**
    - `Billing inquiry` / `Refund request` $\rightarrow$ **Billing**
    - `Cancellation request` $\rightarrow$ **Retention**
    - `Product inquiry` $\rightarrow$ **Product**

- **Semantic Vector Search (Similar Complaints)**:
  - Generates dense sentence embeddings using Hugging Face's `all-MiniLM-L6-v2`.
  - Computes cosine similarity across pre-indexed ticket vectors to retrieve the Top-$N$ most relevant historical tickets with similarity scores.

- **RESTful API & Responsive UI**:
  - Clean web dashboard (`templates/index.html`, `static/css/style.css`, `static/js/app.js`).
  - Standard JSON endpoints (`/predict`, `/similar`, `/analyze`) for seamless integration with existing ticketing systems (Zendesk, Freshdesk, Jira, Salesforce).

---

## 🏗 System Architecture

```text
+--------------------------------------------------------------------+
|                         ResolveAI UI                               |
|        (Browser Dashboard / AJAX Fetch API / Glassmorphism)         |
+---------------------------------+----------------------------------+
                                  |
                           HTTP POST Requests
                                  |
                                  v
+--------------------------------------------------------------------+
|                          Flask API Server                          |
|                             (app.py)                               |
+-------------------+-----------------------------+------------------+
                    |                             |
                    v                             v
     +------------------------------+  +-------------------------------+
     |     Prediction Engine        |  |   Semantic Search Engine      |
     |   (src/prediction.py)        |  |    (src/prediction.py)        |
     +--------------+---------------+  +---------------+---------------+
                    |                                  |
                    | Rule-based Patterns              | Dense Vectors
                    | + Scikit-Learn Models            | Sentence-Transformers
                    v                                  v
     +------------------------------+  +-------------------------------+
     |      Model Artifacts         |  |      Embedding Store          |
     | • category_model.pkl         |  | • complaint_embeddings.pkl   |
     | • priority_model.pkl         |  | • complaints_data.pkl         |
     | • sentiment_model.pkl        |  | • all-MiniLM-L6-v2 model      |
     | • emotion_model.pkl          |  +-------------------------------+
     +------------------------------+
```

---

## 📁 Project Structure

```text
customer_complaint_intelligence/
├── .gitignore                      # Git ignore patterns (virtual env, pycache, OS files)
├── README.md                       # Comprehensive project documentation
├── app.py                          # Flask application server & routing endpoints
├── requirements.txt                # Python package dependencies
│
├── datasets/                       # Datasets used for analysis & training
│   ├── customer_support_tickets.csv    # Raw source customer support tickets
│   ├── complaints_preprocessed.csv     # Cleaned, tokenized complaints
│   └── final_dataset.csv               # Dataset augmented with sentiment & emotion
│
├── notebooks/                      # Data science and model training notebooks
│   ├── artifacts/                      # Serialized ML models and precomputed embeddings
│   │   ├── category_model.pkl          # Trained category classifier
│   │   ├── priority_model.pkl          # Trained priority classifier
│   │   ├── sentiment_model.pkl         # Trained sentiment classifier
│   │   ├── emotion_model.pkl           # Trained emotion classifier
│   │   ├── complaint_embeddings.pkl    # Precomputed dense vector embeddings
│   │   └── complaints_data.pkl         # Pickled ticket metadata for fast retrieval
│   ├── data_set.ipynb                  # Data inspection & initial exploration
│   ├── eda.ipynb                       # Exploratory Data Analysis & visual insights
│   ├── generate_dataset.py             # Script to build final dataset with VADER & emotion rules
│   ├── preprocessing.ipynb             # Text normalization, cleaning & stopword removal
│   └── training.ipynb                  # Model experimentation, training, evaluation & export
│
├── src/                            # Production source code
│   ├── __init__.py                     # Package initialization
│   ├── load_models.py                  # Lazy loading & caching for models and embeddings
│   └── prediction.py                   # Business logic for inference & similarity search
│
├── static/                         # Frontend static assets
│   ├── css/
│   │   └── style.css                   # Space/glassmorphism dark UI styling
│   └── js/
│       └── app.js                      # Client-side AJAX handling, animations, rendering
│
└── templates/                      # Flask HTML templates
    └── index.html                      # Single-page ResolveAI dashboard
```

---

## 🔬 Machine Learning & NLP Pipeline

1. **Text Preprocessing**:
   - Lowercasing, punctuation stripping, whitespace normalization, and stopword removal.
   - Concatenation of ticket subjects and ticket descriptions into unified complaint representations.

2. **Sentiment & Emotion Labeling**:
   - Sentiment scored using NLTK's **VADER** compound metric (`Positive` $\ge 0.05$, `Negative` $\le -0.05$, otherwise `Neutral`).
   - Granular emotion detection using scored keyword dictionaries for `Angry`, `Frustrated`, `Sad`, and `Satisfied`.

3. **Classification Models**:
   - Feature Extraction: **TF-IDF Vectorizer** (with unigrams and bigrams).
   - Classifiers Evaluated: **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Classifier (LinearSVC)**.
   - Trained pipelines packaged via `joblib` into standalone `.pkl` artifacts.

4. **Semantic Embedding & Vector Search**:
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense representations).
   - Fast cosine similarity calculation against precomputed embeddings of indexed tickets.

---

## 🛠 Getting Started

### Prerequisites

- **Python 3.10+** (Python 3.11 or 3.13 recommended)
- **pip** package manager
- Internet access on first run (to download the `all-MiniLM-L6-v2` weights if not cached)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/customer_complaint_intelligence.git
   cd customer_complaint_intelligence
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv myenv
     .\myenv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv myenv
     source myenv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Launch the Flask development server**:
   ```bash
   python app.py
   ```
2. **Access the application**:
   Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```

---

## 🔌 API Reference

### 1. Predict Complaint Attributes
Classifies category, priority, sentiment, emotion, and department.

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json` (or `application/x-www-form-urlencoded`)

**Request Payload:**
```json
{
  "complaint_text": "I was charged twice on my credit card for the monthly subscription. Please fix this immediately!"
}
```

**Response (200 OK):**
```json
{
  "category": "Billing inquiry",
  "priority": "High",
  "sentiment": "Negative",
  "emotion": "Frustrated",
  "department": "Billing"
}
```

---

### 2. Find Similar Complaints
Finds the top-$N$ semantically similar complaints using cosine similarity.

- **URL**: `/similar`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Payload:**
```json
{
  "complaint_text": "My software keeps crashing upon launch after the latest update.",
  "top_n": 3
}
```

**Response (200 OK):**
```json
{
  "similar_complaints": [
    {
      "complaint_text": "Software crash. Application freezes and shuts down when opened...",
      "category": "Technical issue",
      "priority": "High",
      "similarity_score": 0.842
    },
    {
      "complaint_text": "Update failure. The system stopped working right after upgrading...",
      "category": "Technical issue",
      "priority": "Medium",
      "similarity_score": 0.791
    }
  ]
}
```

---

### 3. Full Complaint Analysis
Runs both prediction and semantic similarity search in a single call.

- **URL**: `/analyze`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Payload:**
```json
{
  "complaint_text": "I ordered a replacement battery two weeks ago and it has not arrived yet.",
  "top_n": 5
}
```

**Response (200 OK):**
```json
{
  "prediction": {
    "category": "Product inquiry",
    "priority": "Medium",
    "sentiment": "Negative",
    "emotion": "Not Satisfied",
    "department": "Product"
  },
  "similar_complaints": [ ... ]
}
```

---

## 📓 Jupyter Notebooks & Workflow

| Notebook / Script | Description |
|-------------------|-------------|
| `notebooks/data_set.ipynb` | Initial inspection of the raw customer support dataset. |
| `notebooks/generate_dataset.py` | Fast generation of `final_dataset.csv` with automated VADER sentiment and keyword-based emotion tagging. |
| `notebooks/eda.ipynb` | Comprehensive EDA covering category frequencies, priority breakdowns, sentiment distributions, and text length stats. |
| `notebooks/preprocessing.ipynb` | Text cleaning pipeline: regex normalization, tokenization, stopword filtering. |
| `notebooks/training.ipynb` | TF-IDF feature extraction, model benchmarking (Naive Bayes, Logistic Regression, Linear SVM), metric evaluation, and export to `notebooks/artifacts/`. |

---

## 💻 Tech Stack

- **Backend**: Python, Flask
- **Machine Learning & NLP**: Scikit-Learn, Sentence-Transformers, PyTorch, NLTK, VADER
- **Vector Search**: Dense Cosine Similarity / ChromaDB-ready
- **Data Engineering**: Pandas, NumPy
- **Frontend**: HTML5, Modern CSS3 (Variables, Flexbox/Grid, Glassmorphism), Vanilla JavaScript (ES6+ Fetch API)
- **Model Storage**: Joblib (`.pkl`)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
