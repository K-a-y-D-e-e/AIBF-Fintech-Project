import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraud_detection import FraudDetector


def test_score_transaction_returns_expected_keys():
    detector = FraudDetector()
    txn = {
        "amount": 50.0, "hour_of_day": 14, "txn_per_hour": 1,
        "distance_from_home_km": 2.0, "is_foreign": 0,
    }
    result = detector.score_transaction(txn)
    assert "flagged_as_fraud" in result
    assert "anomaly_score" in result
    assert "risk_factors" in result
    assert isinstance(result["flagged_as_fraud"], bool)


def test_extreme_transaction_more_likely_flagged():
    detector = FraudDetector()
    normal_txn = {
        "amount": 40.0, "hour_of_day": 13, "txn_per_hour": 1,
        "distance_from_home_km": 3.0, "is_foreign": 0,
    }
    suspicious_txn = {
        "amount": 5000.0, "hour_of_day": 3, "txn_per_hour": 12,
        "distance_from_home_km": 800.0, "is_foreign": 1,
    }
    normal_result = detector.score_transaction(normal_txn)
    suspicious_result = detector.score_transaction(suspicious_txn)
    # anomaly_score: lower = more anomalous
    assert suspicious_result["anomaly_score"] < normal_result["anomaly_score"]
