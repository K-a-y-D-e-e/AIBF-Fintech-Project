"""
chatbot.py
----------
AI-based Banking Assistant Chatbot. Uses TF-IDF + cosine similarity
intent classification to understand free-text user queries and route
them to the appropriate module (credit scoring, fraud check, expense
categorization, robo-advisor) or answer general FAQs.

This is intentionally lightweight (no external LLM API dependency) so
the whole project can run fully offline/self-contained.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTENTS = {
    "credit_check": [
        "am i eligible for a loan", "check my credit risk", "will my loan be approved",
        "can i get a loan", "loan eligibility", "credit score check",
    ],
    "fraud_check": [
        "is this transaction fraud", "check for fraud", "suspicious transaction",
        "was i scammed", "flag this payment", "report unauthorized transaction",
    ],
    "expense_help": [
        "categorize my expenses", "where did i spend money", "monthly spending summary",
        "how much did i spend on food", "budget breakdown", "track my expenses",
    ],
    "investment_advice": [
        "how should i invest", "recommend a portfolio", "investment advice",
        "where to invest my savings", "risk profile for investing", "robo advisor",
    ],
    "greeting": [
        "hi", "hello", "hey there", "good morning", "good evening",
    ],
    "help": [
        "what can you do", "help", "list features", "how does this work",
    ],
}

FAQ_RESPONSES = {
    "greeting": "Hello! I'm your AI banking assistant. I can help with loan eligibility, "
                "fraud checks, expense tracking, and investment advice.",
    "help": "I can: (1) check loan/credit eligibility, (2) screen a transaction for fraud, "
            "(3) categorize your expenses, and (4) recommend an investment portfolio. "
            "Just describe what you need in plain language.",
}


class BankingChatbot:
    def __init__(self):
        self._labels = []
        self._examples = []
        for intent, phrases in INTENTS.items():
            for p in phrases:
                self._labels.append(intent)
                self._examples.append(p)

        self.vectorizer = TfidfVectorizer()
        self._example_vectors = self.vectorizer.fit_transform(self._examples)

    def detect_intent(self, message: str, threshold: float = 0.15) -> dict:
        msg_vector = self.vectorizer.transform([message.lower()])
        sims = cosine_similarity(msg_vector, self._example_vectors)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < threshold:
            return {"intent": "unknown", "confidence": round(best_score, 3)}

        return {"intent": self._labels[best_idx], "confidence": round(best_score, 3)}

    def respond(self, message: str) -> dict:
        result = self.detect_intent(message)
        intent = result["intent"]

        if intent in FAQ_RESPONSES:
            reply = FAQ_RESPONSES[intent]
        elif intent == "credit_check":
            reply = ("To check loan eligibility, please provide: age, annual income, "
                      "loan amount, credit history (years), existing debt, employment "
                      "(years), and number of dependents. I'll route this to the Credit "
                      "Risk Scoring module.")
        elif intent == "fraud_check":
            reply = ("To screen a transaction, please share: amount, hour of day, "
                      "recent transaction frequency, distance from home, and whether "
                      "it's a foreign transaction. I'll route this to the Fraud "
                      "Detection module.")
        elif intent == "expense_help":
            reply = ("Share a transaction description (e.g. 'SWIGGY ORDER') and I'll "
                      "categorize it, or upload a statement CSV for a full monthly "
                      "spending summary via the Expense Categorizer module.")
        elif intent == "investment_advice":
            reply = ("To recommend a portfolio, tell me your age, investment horizon, "
                      "how much of your income you can save monthly, and your comfort "
                      "with risk. I'll route this to the Robo-Advisor module.")
        else:
            reply = ("I'm not sure I understood that. I can help with loan eligibility, "
                      "fraud checks, expense categorization, or investment advice.")

        return {"intent": intent, "confidence": result["confidence"], "reply": reply}


if __name__ == "__main__":
    bot = BankingChatbot()
    for msg in ["Hi there", "Can I get a loan?", "Is this payment suspicious?",
                "Where did my money go this month?", "How should I invest 10000 rupees?"]:
        print(f"USER: {msg}\nBOT: {bot.respond(msg)}\n")
