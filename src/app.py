"""
app.py
------
Flask REST API that exposes all AI FinTech modules as HTTP endpoints:

    GET  /health
    POST /api/credit-score       -> Credit Risk Scoring
    POST /api/fraud-check        -> Fraud Detection
    POST /api/categorize-expense -> Expense Categorizer
    POST /api/invest-recommend   -> Robo-Advisor
    POST /api/chat               -> Chatbot (routes to the above)
"""

import os
from flask import Flask, request, jsonify, render_template

from credit_scoring import CreditRiskScorer
from fraud_detection import FraudDetector
from expense_categorizer import ExpenseCategorizer
from robo_advisor import RoboAdvisor
from chatbot import BankingChatbot

app = Flask(__name__)

credit_scorer = CreditRiskScorer()
fraud_detector = FraudDetector()
expense_categorizer = ExpenseCategorizer()
robo_advisor = RoboAdvisor()
chatbot = BankingChatbot()


def _models_ready():
    """Check the trained model artifacts exist before serving predictions."""
    paths = [credit_scorer.model_path, fraud_detector.model_path,
              fraud_detector.scaler_path, expense_categorizer.model_path]
    return all(os.path.exists(p) for p in paths)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_trained": _models_ready()})


@app.route("/api/credit-score", methods=["POST"])
def credit_score():
    data = request.get_json(force=True)
    required = ["age", "annual_income", "loan_amount", "credit_history_years",
                "existing_debt", "employment_years", "num_dependents"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        result = credit_scorer.predict(data)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run training scripts first."}), 503


@app.route("/api/fraud-check", methods=["POST"])
def fraud_check():
    data = request.get_json(force=True)
    required = ["amount", "hour_of_day", "txn_per_hour", "distance_from_home_km", "is_foreign"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        result = fraud_detector.score_transaction(data)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run training scripts first."}), 503


@app.route("/api/categorize-expense", methods=["POST"])
def categorize_expense():
    data = request.get_json(force=True)
    if "description" not in data:
        return jsonify({"error": "Missing field: description"}), 400
    try:
        result = expense_categorizer.categorize(data["description"])
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run training scripts first."}), 503


@app.route("/api/invest-recommend", methods=["POST"])
def invest_recommend():
    data = request.get_json(force=True)
    required = ["age", "investment_horizon_years", "monthly_surplus_pct",
                "loss_tolerance", "investment_experience"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    result = robo_advisor.recommend_portfolio(data)
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    if "message" not in data:
        return jsonify({"error": "Missing field: message"}), 400
    result = chatbot.respond(data["message"])
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
