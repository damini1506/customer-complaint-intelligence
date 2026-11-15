
from flask import Flask, request, jsonify, render_template

from src.prediction import (
    predict_complaint,
    find_similar_complaints
)

app = Flask(__name__)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def index():

    return render_template("index.html")


# ==========================================
# REQUEST DATA
# ==========================================

def get_request_data():

    if request.is_json:

        return request.get_json(silent=True) or {}

    return request.form.to_dict() if request.form else {}


# ==========================================
# CATEGORY / PRIORITY / SENTIMENT PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    data = get_request_data()

    if not data or "complaint_text" not in data:

        return jsonify({
            "error": "complaint_text is required"
        }), 400

    text = str(data["complaint_text"]).strip()

    if not text:

        return jsonify({
            "error": "complaint_text cannot be empty"
        }), 400

    result = predict_complaint(text)

    return jsonify(result)


# ==========================================
# SIMILAR COMPLAINT SEARCH
# ==========================================

@app.route("/similar", methods=["POST"])
def similar():

    data = get_request_data()

    if not data or "complaint_text" not in data:

        return jsonify({
            "error": "complaint_text is required"
        }), 400

    text = str(data["complaint_text"]).strip()

    if not text:

        return jsonify({
            "error": "complaint_text cannot be empty"
        }), 400

    try:

        top_n = int(data.get("top_n", 5))

    except (ValueError, TypeError):

        top_n = 5

    results = find_similar_complaints(
        text,
        top_n=top_n
    )

    return jsonify({
        "similar_complaints": results
    })


# ==========================================
# FULL COMPLAINT ANALYSIS
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze():

    data = get_request_data()

    if not data or "complaint_text" not in data:

        return jsonify({
            "error": "complaint_text is required"
        }), 400

    text = str(data["complaint_text"]).strip()

    if not text:

        return jsonify({
            "error": "complaint_text cannot be empty"
        }), 400

    try:

        top_n = int(data.get("top_n", 5))

    except (ValueError, TypeError):

        top_n = 5

    prediction = predict_complaint(text)

    similar = find_similar_complaints(
        text,
        top_n=top_n
    )

    return jsonify({

        "prediction": prediction,

        "similar_complaints": similar

    })


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )
