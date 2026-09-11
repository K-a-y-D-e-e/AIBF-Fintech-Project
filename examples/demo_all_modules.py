"""
demo_all_modules.py
--------------------
End-to-end demo that exercises every AI module directly (no HTTP layer
needed). Run this after training models to see a full walkthrough of
a simulated customer journey:

    1. A new customer applies for a loan          -> Credit Risk Scoring
    2. Their card is used for an odd transaction   -> Fraud Detection
    3. They ask the chatbot to explain their spend -> Chatbot + Expense Categorizer
    4. They ask for investment advice              -> Robo-Advisor

Run with:  python examples/demo_all_modules.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from credit_scoring import CreditRiskScorer
from fraud_detection import FraudDetector
from expense_categorizer import ExpenseCategorizer
from robo_advisor import RoboAdvisor
from chatbot import BankingChatbot


def main():
    print("=" * 60)
    print("AI FINTECH SUITE - END TO END DEMO")
    print("=" * 60)

    bot = BankingChatbot()

    # 1. Loan application
    print("\n[1] Customer: 'Can I get a loan?'")
    print("Bot:", bot.respond("Can I get a loan?")["reply"])
    applicant = {
        "age": 33, "annual_income": 72000, "loan_amount": 18000,
        "credit_history_years": 9, "existing_debt": 5000,
        "employment_years": 6, "num_dependents": 2,
    }
    scorer = CreditRiskScorer()
    print("Credit Risk Scoring result:", scorer.predict(applicant))

    # 2. Suspicious transaction
    print("\n[2] Customer: 'Is this payment suspicious?'")
    print("Bot:", bot.respond("Is this payment suspicious?")["reply"])
    txn = {
        "amount": 3899.50, "hour_of_day": 3, "txn_per_hour": 7,
        "distance_from_home_km": 620.0, "is_foreign": 1,
    }
    detector = FraudDetector()
    print("Fraud Detection result:", detector.score_transaction(txn))

    # 3. Expense categorization
    print("\n[3] Customer: 'Where did my money go this month?'")
    print("Bot:", bot.respond("Where did my money go this month?")["reply"])
    categorizer = ExpenseCategorizer()
    for desc in ["SWIGGY ORDER", "UBER TRIP FARE", "MONTHLY RENT TRANSFER"]:
        print("  ->", categorizer.categorize(desc))

    # 4. Investment advice
    print("\n[4] Customer: 'How should I invest my savings?'")
    print("Bot:", bot.respond("How should I invest my savings?")["reply"])
    profile = {
        "age": 33, "investment_horizon_years": 12,
        "monthly_surplus_pct": 25, "loss_tolerance": 0.55,
        "investment_experience": 0.3,
    }
    advisor = RoboAdvisor()
    print("Robo-Advisor result:", advisor.recommend_portfolio(profile))

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
