"""
credit_scoring.py
------------------
AI-based Credit Risk Scoring module.

Trains a RandomForestClassifier on applicant financial attributes to
predict the probability that an applicant is a HIGH credit risk
(i.e., likely to default). Exposes a simple class-based API so it can
be reused by the Flask app, the chatbot, or any other module.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "credit_model.joblib")
FEATURES = [
    "age", "annual_income", "loan_amount", "credit_history_years",
    "existing_debt", "employment_years", "num_dependents",
]


class CreditRiskScorer:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None

    def train(self, csv_path: str, save: bool = True) -> dict:
        df = pd.read_csv(csv_path)
        X = df[FEATURES]
        y = df["default_risk"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
            "report": classification_report(y_test, preds, output_dict=True),
        }

        if save:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)

        return metrics

    def load(self):
        if self.model is None:
            self.model = joblib.load(self.model_path)
        return self.model

    def predict(self, applicant: dict) -> dict:
        """
        applicant: dict with keys matching FEATURES
        returns: risk label + probability + a plain-language explanation
        """
        self.load()
        row = pd.DataFrame([applicant])[FEATURES]
        prob_high_risk = float(self.model.predict_proba(row)[0][1])
        label = "HIGH_RISK" if prob_high_risk >= 0.5 else "LOW_RISK"

        decision = "REJECT" if prob_high_risk >= 0.6 else (
            "MANUAL_REVIEW" if prob_high_risk >= 0.4 else "APPROVE"
        )

        return {
            "risk_label": label,
            "default_probability": round(prob_high_risk, 4),
            "decision": decision,
        }

    def feature_importance(self) -> dict:
        self.load()
        return dict(sorted(
            zip(FEATURES, self.model.feature_importances_.round(4)),
            key=lambda x: -x[1]
        ))


if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname(__file__), "..", "dataset", "loan_applicants.csv")
    scorer = CreditRiskScorer()
    metrics = scorer.train(dataset)
    print("Training complete.")
    print("Accuracy:", metrics["accuracy"], "| ROC-AUC:", metrics["roc_auc"])
    print("Feature importance:", scorer.feature_importance())

    sample = {
        "age": 29, "annual_income": 42000, "loan_amount": 25000,
        "credit_history_years": 3, "existing_debt": 15000,
        "employment_years": 2, "num_dependents": 1,
    }
    print("Sample prediction:", scorer.predict(sample))
