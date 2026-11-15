"""
generate_dataset.py
-------------------
Fast dataset generation for customer complaint intelligence.

Uses VADER (NLTK) for sentiment and keyword-based emotion detection.
Processes all 8,469 rows in under 10 seconds on any machine -- no GPU needed.

Run:  python generate_dataset.py
"""

import os
import re
import pandas as pd
from tqdm import tqdm

# --- Configuration -----------------------------------------------------------
CSV_INPUT  = os.path.join(os.path.dirname(__file__), "..", "datasets", "customer_support_tickets.csv")
CSV_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "datasets", "final_dataset.csv")


# --- 1. Load data -----------------------------------------------------------
print("\n[1/5] Loading CSV...")
df = pd.read_csv(CSV_INPUT)
print(f"      Loaded {len(df):,} rows x {len(df.columns)} columns")


# --- 2. Build cleaned DataFrame ----------------------------------------------
print("[2/5] Building cleaned DataFrame...")
new_df = pd.DataFrame()
new_df["complaint_id"]   = df["Ticket ID"]
new_df["complaint_text"] = (
    df["Ticket Subject"].fillna("") + ". " +
    df["Ticket Description"].fillna("")
)
new_df["category"] = df["Ticket Type"]
new_df["priority"] = df["Ticket Priority"]

texts = new_df["complaint_text"].tolist()


# --- 3. Sentiment analysis (VADER - instant) ---------------------------------
print("[3/5] Running sentiment analysis (VADER)...")

import nltk
# Download vader_lexicon if not already present (tiny, <1 MB)
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Classify text as Positive / Negative / Neutral using VADER compound score."""
    score = sid.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

tqdm.pandas(desc="      Sentiment")
new_df["sentiment"] = new_df["complaint_text"].progress_apply(get_sentiment)


# --- 4. Emotion classification (keyword-based - instant) ---------------------
print("[4/5] Running emotion classification...")

# Keyword lists for emotion detection
EMOTION_KEYWORDS = {
    "Angry": [
        "angry", "furious", "outraged", "unacceptable", "terrible",
        "worst", "horrible", "disgusting", "pathetic", "ridiculous",
        "absurd", "infuriating", "livid", "enraged", "mad",
        "hate", "despise", "appalling", "atrocious", "outrageous"
    ],
    "Frustrated": [
        "frustrated", "annoyed", "irritated", "disappointing", "fed up",
        "sick of", "tired of", "useless", "waste", "pointless",
        "unhelpful", "incompetent", "runaround", "still not", "again and again",
        "multiple times", "no response", "no reply", "ignored", "keeps happening",
        "broken again", "not working again", "same issue", "same problem"
    ],
    "Sad": [
        "sad", "disappointed", "upset", "heartbroken", "let down",
        "regret", "unfortunate", "unhappy", "depressed", "miserable",
        "grief", "sorry", "devastated", "disheartened", "hopeless"
    ],
    "Satisfied": [
        "happy", "satisfied", "great", "excellent", "wonderful",
        "fantastic", "love", "thank", "thanks", "appreciate",
        "grateful", "pleased", "awesome", "perfect", "amazing",
        "good job", "well done", "helpful", "resolved", "fixed"
    ],
}

def get_emotion(text):
    """Classify emotion based on keyword matching with scoring."""
    text_lower = str(text).lower()
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[emotion] = score

    if scores:
        return max(scores, key=scores.get)
    return "Not Satisfied"

tqdm.pandas(desc="      Emotion  ")
new_df["emotion"] = new_df["complaint_text"].progress_apply(get_emotion)


# --- 4b. Add 'department' column (mapped from category) ----------------------
print("      Mapping departments...")

DEPARTMENT_MAP = {
    "Technical issue":      "Technical Support",
    "Billing inquiry":      "Billing",
    "Product inquiry":      "Product",
    "Cancellation request": "Retention",
    "Refund request":       "Billing",
}
new_df["department"] = new_df["category"].map(DEPARTMENT_MAP).fillna("General Support")


# --- 5. Export ----------------------------------------------------------------
print("[5/5] Exporting final dataset...")
final_df = new_df[[
    "complaint_id",
    "complaint_text",
    "category",
    "priority",
    "sentiment",
    "emotion",
    "department",
]].copy()

final_df.to_csv(CSV_OUTPUT, index=False)

print(f"\nDone! Saved {len(final_df):,} rows to:\n  {os.path.abspath(CSV_OUTPUT)}")
print(f"\nSentiments : {final_df['sentiment'].unique().tolist()}")
print(f"Emotions   : {final_df['emotion'].unique().tolist()}")
print(f"Departments: {final_df['department'].unique().tolist()}")
print(f"\nDistribution:")
print(f"  Sentiment:\n{final_df['sentiment'].value_counts().to_string()}")
print(f"\n  Emotion:\n{final_df['emotion'].value_counts().to_string()}")
print(f"\n  Department:\n{final_df['department'].value_counts().to_string()}")
print(f"\nNulls:\n{final_df.isnull().sum()}")
