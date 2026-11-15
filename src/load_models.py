import os
import joblib

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks", "artifacts"
)

category_model = joblib.load(
    os.path.join(ARTIFACTS_DIR, "category_model.pkl")
)

priority_model = joblib.load(
    os.path.join(ARTIFACTS_DIR, "priority_model.pkl")
)

sentiment_model = joblib.load(
    os.path.join(ARTIFACTS_DIR, "sentiment_model.pkl")
)

emotion_model = joblib.load(
    os.path.join(ARTIFACTS_DIR, "emotion_model.pkl")
)



complaint_embeddings = joblib.load(
    os.path.join(ARTIFACTS_DIR, "complaint_embeddings.pkl")
)



try:
    complaints_df = joblib.load(
        os.path.join(ARTIFACTS_DIR, "complaints_data.pkl")
    )
except Exception:
    import pandas as pd
    complaints_df = pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "complaints_preprocessed.csv")
    )



from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

department_mapping = {
    "Technical issue": "Technical Support",
    "Billing inquiry": "Billing",
    "Product inquiry": "Product",
    "Cancellation request": "Retention",
    "Refund request": "Billing",
}

print("All models and artifacts loaded successfully!")
