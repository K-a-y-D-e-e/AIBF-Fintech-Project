# API Reference

Base URL (local): `http://127.0.0.1:5000`

All endpoints accept and return `application/json`.

---

## `GET /health`

Health check. Also reports whether trained model artifacts are present.

**Response**
```json
{ "status": "ok", "models_trained": true }
```

---

## `POST /api/credit-score`

Predicts credit/default risk for a loan applicant.

**Request body**
```json
{
  "age": 29,
  "annual_income": 42000,
  "loan_amount": 25000,
  "credit_history_years": 3,
  "existing_debt": 15000,
  "employment_years": 2,
  "num_dependents": 1
}
```

**Response**
```json
{
  "risk_label": "HIGH_RISK",
  "default_probability": 0.9398,
  "decision": "REJECT"
}
```
`decision` is one of `APPROVE`, `MANUAL_REVIEW`, `REJECT`.

---

## `POST /api/fraud-check`

Screens a transaction for anomalous/fraudulent behaviour.

**Request body**
```json
{
  "amount": 4200,
  "hour_of_day": 2,
  "txn_per_hour": 8,
  "distance_from_home_km": 550,
  "is_foreign": 1
}
```

**Response**
```json
{
  "flagged_as_fraud": true,
  "anomaly_score": -0.1173,
  "risk_factors": [
    "unusually high transaction amount",
    "transaction occurred at an unusual hour",
    "transaction location far from usual activity area",
    "foreign transaction",
    "high transaction frequency in short time window"
  ]
}
```

---

## `POST /api/categorize-expense`

Classifies a raw transaction description into a spending category.

**Request body**
```json
{ "description": "NETFLIX SUBSCRIPTION" }
```

**Response**
```json
{
  "description": "NETFLIX SUBSCRIPTION",
  "predicted_category": "Entertainment",
  "confidence": 0.9329
}
```

---

## `POST /api/invest-recommend`

Recommends a portfolio allocation based on a risk-tolerance profile.

**Request body**
```json
{
  "age": 27,
  "investment_horizon_years": 15,
  "monthly_surplus_pct": 30,
  "loss_tolerance": 0.7,
  "investment_experience": 0.4
}
```

**Response**
```json
{
  "risk_score": 61.75,
  "risk_band": "Growth",
  "recommended_allocation": { "Bonds/Debt": 20, "Equity": 65, "Gold": 10, "Cash": 5 },
  "rationale": "Based on a risk-tolerance score of 61.75/100 ..."
}
```

---

## `POST /api/chat`

Natural-language entry point. Detects intent and routes to the
relevant module's explanation, or answers directly for FAQs/greetings.

**Request body**
```json
{ "message": "Can I get a loan?" }
```

**Response**
```json
{
  "intent": "credit_check",
  "confidence": 1.0,
  "reply": "To check loan eligibility, please provide: age, annual income, ..."
}
```

---

## Error Responses

- `400 Bad Request` — missing required fields, with a JSON body listing them.
- `503 Service Unavailable` — a model endpoint was called before training (run the training scripts in `src/` first, or build via Docker which trains automatically).
