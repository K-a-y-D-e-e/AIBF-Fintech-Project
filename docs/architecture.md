# Architecture

## Component Diagram

```
                         ┌──────────────────────────────┐
                         │        Client (curl /        │
                         │   Postman / example scripts)  │
                         └───────────────┬───────────────┘
                                         │ HTTP (JSON)
                                         ▼
                         ┌──────────────────────────────┐
                         │        Flask REST API          │
                         │           (src/app.py)          │
                         └───────────────┬───────────────┘
                                         │
           ┌────────────────┬───────────┼───────────┬────────────────┐
           ▼                ▼           ▼           ▼                ▼
 ┌──────────────────┐ ┌────────────┐ ┌─────────────────┐ ┌────────────────┐ ┌───────────────┐
 │ Credit Risk       │ │ Fraud       │ │ Expense          │ │ Robo-Advisor    │ │ Banking        │
 │ Scoring            │ │ Detection   │ │ Categorizer       │ │                 │ │ Chatbot        │
 │ (Random Forest)    │ │ (Isolation  │ │ (TF-IDF + Naive   │ │ (Rule-weighted  │ │ (TF-IDF intent │
 │                    │ │  Forest)    │ │  Bayes)           │ │  risk scoring)  │ │  matching)     │
 └─────────┬──────────┘ └──────┬─────┘ └─────────┬────────┘ └────────┬────────┘ └────────┬───────┘
           │                   │                  │                    │                   │
           ▼                   ▼                  ▼                    │                   │
 ┌───────────────────────────────────────────────────────────┐        │        (routes to any
 │              Trained Model Artifacts (config/*.joblib)       │        │         of the 4 modules)
 └───────────────────────────────────────────────────────────┘        │
           ▲                   ▲                  ▲                    │
           └───────────────────┴──────────────────┴────────────────────┘
                                         │
                         ┌──────────────────────────────┐
                         │   Synthetic Data Generator     │
                         │   (src/utils/data_generator.py)│
                         └──────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │   dataset/*.csv (training data)│
                         └──────────────────────────────┘
```

## Layers

1. **Data Layer** (`src/utils/data_generator.py`, `dataset/`)
   Generates synthetic-but-realistic banking datasets: loan applicants,
   transactions, and expense descriptions. No real customer data is used.

2. **AI/ML Layer** (`src/credit_scoring.py`, `src/fraud_detection.py`,
   `src/expense_categorizer.py`, `src/robo_advisor.py`)
   Each module encapsulates one machine-learning or rule-based AI model
   behind a small, testable class with `train()` and `predict()`-style
   methods. Trained model artifacts are persisted with `joblib` under
   `config/` so they don't need to be retrained on every request.

3. **Orchestration Layer** (`src/chatbot.py`)
   A lightweight NLP intent classifier that interprets free-text user
   requests and routes them to the correct downstream module — this is
   the "agentic" glue that makes the suite usable through natural
   language instead of raw API calls.

4. **API Layer** (`src/app.py`)
   A Flask REST API exposing each module as a JSON endpoint, plus a
   unified `/api/chat` endpoint for natural-language interaction.

5. **Delivery** (`Dockerfile`)
   Containerizes the whole application: builds the image, generates
   data, trains all models, and starts the API server.

## Why these AI techniques

| Module | AI Technique | Reasoning |
|---|---|---|
| Credit Risk Scoring | Random Forest (supervised classification) | Handles non-linear relationships between income/debt/history well; gives interpretable feature importances for regulatory explainability. |
| Fraud Detection | Isolation Forest (unsupervised anomaly detection) | Fraud patterns evolve and labeled fraud data is scarce/imbalanced in the real world; anomaly detection doesn't require exhaustive fraud labels. |
| Expense Categorizer | TF-IDF + Multinomial Naive Bayes (NLP text classification) | Bank statement descriptions are short, noisy text — a classic text-classification problem where Naive Bayes is fast and effective. |
| Robo-Advisor | Weighted rule-based scoring model | Transparent, auditable risk profiling — mirrors how real robo-advisors justify allocation decisions to regulators and customers. |
| Chatbot | TF-IDF + cosine similarity intent matching | Lightweight, fully offline NLP routing without needing an external LLM API dependency. |

## Data Flow (Example: Fraud Check)

1. Client sends `POST /api/fraud-check` with transaction attributes.
2. `app.py` validates required fields and forwards to `FraudDetector`.
3. `FraudDetector` loads the trained `IsolationForest` + `StandardScaler`
   from `config/`, scales the input, and computes an anomaly score.
4. If flagged, rule-based rationale (`risk_factors`) is generated for
   human-readable explainability.
5. JSON response returned to client.
