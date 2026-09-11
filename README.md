# AI FinTech Suite

An integrated FinTech application that applies Artificial Intelligence
across five core banking/finance workflows: **credit risk scoring**,
**fraud detection**, **expense categorization**, **robo-advisory**, and
a **conversational assistant** that ties them together.

Built as a CA-2 coding assignment demonstrating practical AI integration
in the banking and finance domain (Unit 3 & Unit 4 concepts: credit
scoring, fraud analytics, digital/robo advisory, and NLP-based banking
assistants).

---

## Features / Modules

| Module | File | AI Technique | What it does |
|---|---|---|---|
| Credit Risk Scoring | `src/credit_scoring.py` | Random Forest (supervised ML) | Predicts loan default risk and gives an approve/review/reject decision |
| Fraud Detection | `src/fraud_detection.py` | Isolation Forest (anomaly detection) | Flags suspicious transactions in real time with a rationale |
| Expense Categorizer | `src/expense_categorizer.py` | TF-IDF + Naive Bayes (NLP) | Auto-tags bank statement line items into spending categories |
| Robo-Advisor | `src/robo_advisor.py` | Weighted risk-profiling model | Recommends a diversified portfolio allocation from an investor questionnaire |
| Banking Chatbot | `src/chatbot.py` | TF-IDF + cosine similarity (intent classification) | Understands free-text queries and routes them to the right module |
| REST API | `src/app.py` | Flask | Exposes all modules as HTTP JSON endpoints |

See [`docs/architecture.md`](docs/architecture.md) for the full system
diagram and [`docs/api_reference.md`](docs/api_reference.md) for
endpoint details.

---

## Project Structure

```
ai-fintech-suite/
├── src/
│   ├── app.py                   # Flask REST API
│   ├── credit_scoring.py        # Credit risk module
│   ├── fraud_detection.py       # Fraud detection module
│   ├── expense_categorizer.py   # Expense categorization module
│   ├── robo_advisor.py          # Robo-advisor module
│   ├── chatbot.py               # Chatbot / intent router
│   └── utils/
│       └── data_generator.py    # Synthetic dataset generator
├── tests/                       # Pytest unit tests for every module
├── examples/                    # Runnable demo scripts (CLI + API client)
├── dataset/                     # Generated synthetic CSV datasets
├── config/                      # config.yaml + trained model artifacts (.joblib)
├── docs/                        # Architecture & API documentation
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Getting Started

### 1. Setup (local, without Docker)

```bash
git clone <this-repo-url>
cd ai-fintech-suite
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python src/utils/data_generator.py
```
This creates `dataset/loan_applicants.csv`, `dataset/transactions.csv`,
and `dataset/expenses.csv`.

### 3. Train the models

```bash
python src/credit_scoring.py
python src/fraud_detection.py
python src/expense_categorizer.py
```
Each script trains its model, prints evaluation metrics, and saves the
trained artifact to `config/`.

### 4. Run the API server

```bash
python src/app.py
```
The API starts on `http://127.0.0.1:5000`.

### 5. Try it out

```bash
python examples/demo_all_modules.py     # runs every module directly, no server needed
python examples/api_client_demo.py      # calls the running Flask API over HTTP
```

Or with `curl`:
```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can I get a loan?"}'
```

### 6. Run tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Run with Docker

```bash
docker build -t ai-fintech-suite .
docker run -p 5000:5000 ai-fintech-suite
```
The image generates data and trains all models automatically at build
time, so the API is ready to use immediately on `http://localhost:5000`.

---

## Notes on the Data

All datasets are **synthetically generated** (`src/utils/data_generator.py`)
using randomized, parameterized distributions that mimic realistic
banking patterns (income, debt, transaction amounts, spend categories).
**No real customer or banking data is used anywhere in this project.**

---

## Disclaimer

This is an academic prototype built for learning purposes. It is not
a certified credit-decisioning, fraud-prevention, or investment-advisory
system and should not be used for real financial decisions.
