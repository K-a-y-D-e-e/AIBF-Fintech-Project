"""
fraud_detection.py
-------------------
AI-based Fraud Detection module using unsupervised anomaly detection
(Isolation Forest). Designed to flag suspicious transactions in
real time without requiring labeled fraud data (though labels, when
available, are used here only for evaluation).
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "fraud_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "fraud_scaler.joblib")
FEATURES = ["amount", "hour_of_day", "txn_per_hour", "distance_from_home_km", "is_foreign"]


class FraudDetector:
    def __init__(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

    def train(self, csv_path: str, contamination: float = 0.03, save: bool = True) -> dict:
        df = pd.read_csv(csv_path)
        X = df[FEATURES]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        )
        self.model.fit(X_scaled)

        # -1 = anomaly (fraud-like), 1 = normal -> convert to 1/0 to compare with labels
        raw_preds = self.model.predict(X_scaled)
        preds = (raw_preds == -1).astype(int)

        report = None
        if "is_fraud" in df.columns:
            report = classification_report(df["is_fraud"], preds, output_dict=True)

        if save:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

        return {"flagged_count": int(preds.sum()), "total": len(df), "report": report}

    def load(self):
        if self.model is None or self.scaler is None:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        return self.model, self.scaler

    def score_transaction(self, txn: dict) -> dict:
        """
        txn: dict with keys matching FEATURES
        returns: fraud flag + anomaly score + rationale
        """
        model, scaler = self.load()
        row = pd.DataFrame([txn])[FEATURES]
        row_scaled = scaler.transform(row)

        raw_pred = model.predict(row_scaled)[0]
        # decision_function: lower (more negative) = more anomalous
        anomaly_score = float(model.decision_function(row_scaled)[0])
        is_flagged = raw_pred == -1

        reasons = []
        if txn["amount"] > 1000:
            reasons.append("unusually high transaction amount")
        if txn["hour_of_day"] < 5 or txn["hour_of_day"] > 22:
            reasons.append("transaction occurred at an unusual hour")
        if txn["distance_from_home_km"] > 100:
            reasons.append("transaction location far from usual activity area")
        if txn.get("is_foreign"):
            reasons.append("foreign transaction")
        if txn["txn_per_hour"] > 5:
            reasons.append("high transaction frequency in short time window")

        return {
            "flagged_as_fraud": bool(is_flagged),
            "anomaly_score": round(anomaly_score, 4),
            "risk_factors": reasons if is_flagged else [],
        }


if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname(__file__), "..", "dataset", "transactions.csv")
    detector = FraudDetector()
    result = detector.train(dataset)
    print("Training complete. Flagged:", result["flagged_count"], "/", result["total"])

    sample_txn = {
        "amount": 4200.0, "hour_of_day": 2, "txn_per_hour": 8,
        "distance_from_home_km": 550.0, "is_foreign": 1,
    }
    print("Sample scoring:", detector.score_transaction(sample_txn))
