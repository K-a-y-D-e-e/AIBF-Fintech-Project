import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from credit_scoring import CreditRiskScorer


def test_predict_returns_expected_keys():
    scorer = CreditRiskScorer()
    applicant = {
        "age": 30, "annual_income": 60000, "loan_amount": 10000,
        "credit_history_years": 8, "existing_debt": 2000,
        "employment_years": 5, "num_dependents": 1,
    }
    result = scorer.predict(applicant)
    assert "risk_label" in result
    assert "default_probability" in result
    assert "decision" in result
    assert result["risk_label"] in ("HIGH_RISK", "LOW_RISK")
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["decision"] in ("APPROVE", "MANUAL_REVIEW", "REJECT")


def test_high_debt_applicant_scores_higher_risk_than_low_debt():
    scorer = CreditRiskScorer()
    low_debt = {
        "age": 35, "annual_income": 90000, "loan_amount": 5000,
        "credit_history_years": 15, "existing_debt": 500,
        "employment_years": 12, "num_dependents": 0,
    }
    high_debt = {
        "age": 35, "annual_income": 40000, "loan_amount": 30000,
        "credit_history_years": 1, "existing_debt": 25000,
        "employment_years": 0, "num_dependents": 3,
    }
    low_result = scorer.predict(low_debt)
    high_result = scorer.predict(high_debt)
    assert high_result["default_probability"] > low_result["default_probability"]
