import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.load_models import (
    category_model,
    priority_model,
    sentiment_model,
    emotion_model,
    department_mapping,
    embedding_model,
    complaint_embeddings,
    complaints_df
)


def classify_category(text):
    text_clean = str(text).lower().strip()

    # 1. Cancellation request intent
    cancellation_patterns = [
        r"\bcancel\b", r"\bcancellation\b", r"\bunsubscribe\b",
        r"\bclose (my )?account\b", r"\bterminate\b", r"\bstop (my )?subscription\b",
        r"\bdeactivate\b", r"\bend (my )?membership\b"
    ]
    if any(re.search(p, text_clean) for p in cancellation_patterns):
        return "Cancellation request"

    # 2. Refund request intent
    refund_patterns = [
        r"\brefund\b", r"\bmoney back\b", r"\breimburse\b", r"\breimbursement\b",
        r"\breturn for (a )?refund\b", r"\bwant my money\b", r"\bget (my )?money back\b"
    ]
    if any(re.search(p, text_clean) for p in refund_patterns):
        return "Refund request"

    # 3. Billing inquiry intent
    billing_patterns = [
        r"\bbill\b", r"\bbilling\b", r"\binvoice\b", r"\bcharged?\b", r"\bovercharged?\b",
        r"\bextra fee\b", r"\bunauthorized charge\b", r"\bcredit card\b", r"\bdebit card\b",
        r"\bpayment (failed|issue|problem|declined)\b", r"\bdoubly charged\b", r"\bcharged twice\b",
        r"\breceipt\b", r"\bpricing\b", r"\bdeducted\b"
    ]
    if any(re.search(p, text_clean) for p in billing_patterns):
        return "Billing inquiry"

    # 4. Technical issue intent
    technical_patterns = [
        r"\bcrash(es|ed|ing)?\b", r"\bbug(s)?\b", r"\bglitch(es)?\b", r"\berror(s)?\b",
        r"\bnot loading\b", r"\bwon'?t load\b", r"\bcan'?t log ?in\b", r"\blogin\b",
        r"\bpassword\b", r"\bserver\b", r"\bconnection\b", r"\bwifi\b", r"\bnetwork\b",
        r"\bsoftware\b", r"\bupdate failed\b", r"\binstall(ation)?\b", r"\bfrozen\b",
        r"\bfreez(e|ing)\b", r"\bblue screen\b", r"\bblack screen\b", r"\bnot responding\b",
        r"\btimeout\b"
    ]
    if any(re.search(p, text_clean) for p in technical_patterns):
        return "Technical issue"

    # 5. Product inquiry intent (damaged, broken, defective, warranty, specs, etc.)
    product_patterns = [
        r"\bdamaged?\b", r"\bbroken\b", r"\bdefect(ive)?\b", r"\bfaulty\b",
        r"\bproduct\b", r"\bitem\b", r"\bpackage\b", r"\bhardware\b",
        r"\bfeatures?\b", r"\bspec(s|ification)?\b", r"\bwarranty\b",
        r"\bquality\b", r"\bcompatibility\b", r"\bhow to use\b", r"\bmanual\b",
        r"\binstructions?\b", r"\bparts?\b", r"\bmissing item\b", r"\bscratch(ed)?\b"
    ]
    if any(re.search(p, text_clean) for p in product_patterns):
        return "Product inquiry"

    # Fallback to trained ML category_model
    return str(category_model.predict([text])[0])


def predict_complaint(text):

    # Category prediction (intent-guided with ML fallback)
    category = classify_category(text)

    # Priority prediction
    priority = str(priority_model.predict([text])[0])

    # Sentiment prediction
    sentiment = str(sentiment_model.predict([text])[0])

    # Emotion prediction (maps Other -> Not Satisfied)
    raw_emotion = str(emotion_model.predict([text])[0])
    emotion = "Not Satisfied" if raw_emotion in ["Other", "Not Satisfied"] else raw_emotion

    # Department prediction
    department = department_mapping.get(category, "General Support")

    return {
        "category": category,
        "priority": priority,
        "sentiment": sentiment,
        "emotion": emotion,
        "department": department
    }


def find_similar_complaints(text, top_n=5):

    query_embedding = embedding_model.encode(
        [text],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        complaint_embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1]

    results = []

    for idx in top_indices:

        results.append({
            "complaint_text": complaints_df.iloc[idx]["complaint_text"],
            "category": complaints_df.iloc[idx]["category"],
            "priority": complaints_df.iloc[idx]["priority"],
            "similarity_score": round(
                float(scores[idx]), 3
            )
        })

        if len(results) == top_n:
            break

    return results
