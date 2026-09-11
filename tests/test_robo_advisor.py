import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from robo_advisor import RoboAdvisor


def test_recommend_portfolio_returns_expected_keys():
    advisor = RoboAdvisor()
    profile = {
        "age": 25, "investment_horizon_years": 20,
        "monthly_surplus_pct": 40, "loss_tolerance": 0.8,
        "investment_experience": 0.5,
    }
    result = advisor.recommend_portfolio(profile)
    assert "risk_score" in result
    assert "risk_band" in result
    assert "recommended_allocation" in result
    assert sum(result["recommended_allocation"].values()) == 100


def test_young_high_tolerance_more_aggressive_than_old_low_tolerance():
    advisor = RoboAdvisor()
    young_aggressive = {
        "age": 24, "investment_horizon_years": 25,
        "monthly_surplus_pct": 45, "loss_tolerance": 0.9,
        "investment_experience": 0.7,
    }
    old_conservative = {
        "age": 58, "investment_horizon_years": 3,
        "monthly_surplus_pct": 10, "loss_tolerance": 0.1,
        "investment_experience": 0.1,
    }
    r1 = advisor.recommend_portfolio(young_aggressive)
    r2 = advisor.recommend_portfolio(old_conservative)
    assert r1["risk_score"] > r2["risk_score"]
