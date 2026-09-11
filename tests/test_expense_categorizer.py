import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from expense_categorizer import ExpenseCategorizer


def test_categorize_returns_expected_keys():
    cat = ExpenseCategorizer()
    result = cat.categorize("SWIGGY ORDER PAYMENT")
    assert "predicted_category" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_categorize_batch_length_matches_input():
    cat = ExpenseCategorizer()
    descriptions = ["NETFLIX SUBSCRIPTION", "ELECTRICITY BILL PAYMENT", "UBER TRIP FARE"]
    results = cat.categorize_batch(descriptions)
    assert len(results) == len(descriptions)
