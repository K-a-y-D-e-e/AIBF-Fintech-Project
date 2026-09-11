"""
data_generator.py
------------------
Generates synthetic, privacy-safe datasets used to train/demo the AI
modules in this project. No real banking data is used anywhere in this
repository.

Datasets produced:
    1. loan_applicants.csv   -> for Credit Risk Scoring module
    2. transactions.csv      -> for Fraud Detection module
    3. expenses.csv          -> for Expense Categorizer module
"""

import os
import random
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")


def generate_loan_applicants(n=1500):
    """Synthetic loan applicant records with a derived 'default_risk' label."""
    age = np.random.randint(21, 65, n)
    income = np.random.normal(55000, 20000, n).clip(12000, 250000)
    loan_amount = np.random.normal(15000, 8000, n).clip(1000, 80000)
    credit_history_years = np.random.randint(0, 30, n)
    existing_debt = np.random.normal(8000, 6000, n).clip(0, 60000)
    employment_years = np.random.randint(0, 35, n)
    num_dependents = np.random.randint(0, 5, n)

    debt_to_income = existing_debt / (income + 1)
    loan_to_income = loan_amount / (income + 1)

    # Latent risk score used only to *simulate* realistic labels
    risk_score = (
        0.45 * debt_to_income
        + 0.35 * loan_to_income
        - 0.02 * credit_history_years
        - 0.01 * employment_years
        + 0.05 * num_dependents
        + np.random.normal(0, 0.15, n)
    )
    threshold = np.quantile(risk_score, 0.72)
    default_risk = (risk_score > threshold).astype(int)  # 1 = high risk

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "loan_amount": loan_amount.round(2),
        "credit_history_years": credit_history_years,
        "existing_debt": existing_debt.round(2),
        "employment_years": employment_years,
        "num_dependents": num_dependents,
        "default_risk": default_risk,
    })
    return df


def generate_transactions(n=4000, fraud_rate=0.03):
    """Synthetic transaction stream with injected anomalies (fraud)."""
    n_fraud = int(n * fraud_rate)
    n_normal = n - n_fraud

    normal = pd.DataFrame({
        "amount": np.random.gamma(shape=2.0, scale=45, size=n_normal).round(2),
        "hour_of_day": np.random.normal(14, 4, n_normal).clip(0, 23).astype(int),
        "txn_per_hour": np.random.poisson(2, n_normal),
        "distance_from_home_km": np.random.exponential(5, n_normal).round(2),
        "is_foreign": np.random.binomial(1, 0.03, n_normal),
        "is_fraud": 0,
    })

    fraud = pd.DataFrame({
        "amount": np.random.gamma(shape=5.0, scale=300, size=n_fraud).round(2),
        "hour_of_day": np.random.choice(list(range(0, 5)) + list(range(22, 24)), n_fraud),
        "txn_per_hour": np.random.poisson(9, n_fraud),
        "distance_from_home_km": np.random.exponential(400, n_fraud).round(2),
        "is_foreign": np.random.binomial(1, 0.6, n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "transaction_id", [f"TXN{i:06d}" for i in range(len(df))])
    return df


def generate_expenses(n=800):
    """Synthetic bank-statement expense descriptions with category labels."""
    templates = {
        "Groceries": ["BIGBAZAAR PURCHASE", "DMART GROCERY", "RELIANCE FRESH", "LOCAL SUPERMARKET BILL"],
        "Dining": ["SWIGGY ORDER", "ZOMATO FOOD DELIVERY", "STARBUCKS COFFEE", "RESTAURANT BILL PAYMENT"],
        "Utilities": ["ELECTRICITY BILL PAYMENT", "WATER BILL AUTOPAY", "BROADBAND RECHARGE", "GAS CYLINDER BOOKING"],
        "Transport": ["UBER TRIP FARE", "OLA CAB PAYMENT", "PETROL PUMP FUEL", "METRO CARD RECHARGE"],
        "Entertainment": ["NETFLIX SUBSCRIPTION", "SPOTIFY PREMIUM", "MOVIE TICKET BOOKING", "GAMING STORE PURCHASE"],
        "Shopping": ["AMAZON ORDER PAYMENT", "FLIPKART PURCHASE", "MYNTRA FASHION BUY", "ELECTRONICS STORE BILL"],
        "Healthcare": ["PHARMACY BILL", "HOSPITAL CONSULTATION FEE", "DIAGNOSTIC LAB TEST", "MEDICAL INSURANCE PREMIUM"],
        "Rent/Housing": ["MONTHLY RENT TRANSFER", "SOCIETY MAINTENANCE FEE", "HOME LOAN EMI", "PROPERTY TAX PAYMENT"],
    }
    rows = []
    for _ in range(n):
        cat = random.choice(list(templates.keys()))
        desc = random.choice(templates[cat])
        amount = round(np.random.gamma(2.0, 40), 2)
        rows.append({"description": desc, "amount": amount, "category": cat})
    return pd.DataFrame(rows)


def generate_all(output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    loans = generate_loan_applicants()
    txns = generate_transactions()
    expenses = generate_expenses()

    loans.to_csv(os.path.join(output_dir, "loan_applicants.csv"), index=False)
    txns.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    expenses.to_csv(os.path.join(output_dir, "expenses.csv"), index=False)

    print(f"Generated: loan_applicants.csv ({len(loans)} rows)")
    print(f"Generated: transactions.csv ({len(txns)} rows)")
    print(f"Generated: expenses.csv ({len(expenses)} rows)")


if __name__ == "__main__":
    generate_all()
