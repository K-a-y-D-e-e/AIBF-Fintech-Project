"""
api_client_demo.py
--------------------
Demonstrates calling the running Flask API using the `requests`
library. Start the server first:

    python src/app.py

Then, in another terminal:

    python examples/api_client_demo.py
"""

import requests

BASE_URL = "http://127.0.0.1:5000"


def main():
    print("Health check:", requests.get(f"{BASE_URL}/health").json())

    credit_payload = {
        "age": 29, "annual_income": 42000, "loan_amount": 25000,
        "credit_history_years": 3, "existing_debt": 15000,
        "employment_years": 2, "num_dependents": 1,
    }
    print("Credit score:", requests.post(f"{BASE_URL}/api/credit-score", json=credit_payload).json())

    fraud_payload = {
        "amount": 4200, "hour_of_day": 2, "txn_per_hour": 8,
        "distance_from_home_km": 550, "is_foreign": 1,
    }
    print("Fraud check:", requests.post(f"{BASE_URL}/api/fraud-check", json=fraud_payload).json())

    expense_payload = {"description": "STARBUCKS COFFEE"}
    print("Expense category:", requests.post(f"{BASE_URL}/api/categorize-expense", json=expense_payload).json())

    invest_payload = {
        "age": 27, "investment_horizon_years": 15, "monthly_surplus_pct": 30,
        "loss_tolerance": 0.7, "investment_experience": 0.4,
    }
    print("Investment advice:", requests.post(f"{BASE_URL}/api/invest-recommend", json=invest_payload).json())

    chat_payload = {"message": "How should I invest 10000 rupees?"}
    print("Chatbot:", requests.post(f"{BASE_URL}/api/chat", json=chat_payload).json())


if __name__ == "__main__":
    main()
