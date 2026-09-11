"""
expense_categorizer.py
-----------------------
AI-based (NLP) Expense Categorizer. Automatically classifies raw bank
statement transaction descriptions into spending categories using
TF-IDF vectorization + Multinomial Naive Bayes text classification.
Used to power automated budgeting/personal-finance insights.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "expense_model.joblib")


class ExpenseCategorizer:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None

    def train(self, csv_path: str, save: bool = True) -> dict:
        df = pd.read_csv(csv_path)
        X_train, X_test, y_train, y_test = train_test_split(
            df["description"], df["category"], test_size=0.2, random_state=42,
            stratify=df["category"]
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
            ("clf", MultinomialNB()),
        ])
        self.pipeline.fit(X_train, y_train)

        preds = self.pipeline.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "report": classification_report(y_test, preds, output_dict=True),
        }

        if save:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.pipeline, self.model_path)

        return metrics

    def load(self):
        if self.pipeline is None:
            self.pipeline = joblib.load(self.model_path)
        return self.pipeline

    def categorize(self, description: str) -> dict:
        self.load()
        category = str(self.pipeline.predict([description])[0])
        proba = self.pipeline.predict_proba([description])[0]
        confidence = float(max(proba))
        return {"description": description, "predicted_category": category, "confidence": round(confidence, 4)}

    def categorize_batch(self, descriptions: list) -> list:
        return [self.categorize(d) for d in descriptions]

    def monthly_summary(self, csv_path: str) -> pd.DataFrame:
        """Categorize an uncategorized expense CSV (needs 'description','amount' cols)
        and return spend totals per predicted category."""
        self.load()
        df = pd.read_csv(csv_path)
        df["predicted_category"] = self.pipeline.predict(df["description"])
        summary = df.groupby("predicted_category")["amount"].sum().sort_values(ascending=False)
        return summary


if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname(__file__), "..", "dataset", "expenses.csv")
    cat = ExpenseCategorizer()
    metrics = cat.train(dataset)
    print("Training complete. Accuracy:", metrics["accuracy"])

    print(cat.categorize("SWIGGY ORDER PAYMENT"))
    print(cat.categorize("ELECTRICITY BILL AUTOPAY"))
