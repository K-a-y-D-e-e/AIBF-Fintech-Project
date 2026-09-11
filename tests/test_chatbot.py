import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chatbot import BankingChatbot


def test_detect_intent_credit_check():
    bot = BankingChatbot()
    result = bot.detect_intent("Can I get a loan approved?")
    assert result["intent"] == "credit_check"


def test_detect_intent_fraud_check():
    bot = BankingChatbot()
    result = bot.detect_intent("I think this transaction is suspicious")
    assert result["intent"] == "fraud_check"


def test_respond_returns_expected_keys():
    bot = BankingChatbot()
    result = bot.respond("How should I invest my savings?")
    assert "intent" in result
    assert "reply" in result
    assert isinstance(result["reply"], str)
    assert len(result["reply"]) > 0
