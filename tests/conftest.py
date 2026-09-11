import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.data_generator import generate_all
from credit_scoring import CreditRiskScorer
from fraud_detection import FraudDetector
from expense_categorizer import ExpenseCategorizer

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


@pytest.fixture(scope="session", autouse=True)
def setup_data_and_models():
    """Ensure datasets and trained models exist before running any test."""
    generate_all(DATASET_DIR)

    CreditRiskScorer().train(os.path.join(DATASET_DIR, "loan_applicants.csv"))
    FraudDetector().train(os.path.join(DATASET_DIR, "transactions.csv"))
    ExpenseCategorizer().train(os.path.join(DATASET_DIR, "expenses.csv"))
    yield
