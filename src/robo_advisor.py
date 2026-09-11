"""
robo_advisor.py
----------------
AI-assisted Robo-Advisory module. Converts a short investor
questionnaire into a quantified risk-tolerance score using a weighted
scoring model, then maps that score to a diversified portfolio
allocation recommendation. This mirrors how real robo-advisors
(e.g., Betterment, Wealthfront) perform initial risk profiling.
"""

RISK_WEIGHTS = {
    "age": lambda v: max(0, (60 - v) / 60) * 20,           # younger -> higher risk capacity
    "investment_horizon_years": lambda v: min(v, 20) / 20 * 25,
    "monthly_surplus_pct": lambda v: min(v, 50) / 50 * 20,  # % of income saved
    "loss_tolerance": lambda v: v * 20,                     # 0-1 self-reported scale
    "investment_experience": lambda v: v * 15,              # 0-1 self-reported scale
}

PORTFOLIOS = {
    "Conservative": {"Bonds/Debt": 65, "Equity": 20, "Gold": 10, "Cash": 5},
    "Moderate":     {"Bonds/Debt": 40, "Equity": 45, "Gold": 10, "Cash": 5},
    "Growth":       {"Bonds/Debt": 20, "Equity": 65, "Gold": 10, "Cash": 5},
    "Aggressive":   {"Bonds/Debt": 5,  "Equity": 85, "Gold": 5,  "Cash": 5},
}


class RoboAdvisor:
    def compute_risk_score(self, profile: dict) -> float:
        """
        profile keys:
            age (int)
            investment_horizon_years (int)
            monthly_surplus_pct (0-100)
            loss_tolerance (0.0-1.0)   # self-reported, from questionnaire
            investment_experience (0.0-1.0)
        returns a 0-100 risk score
        """
        score = 0.0
        for key, fn in RISK_WEIGHTS.items():
            score += fn(profile[key])
        return round(min(score, 100), 2)

    def classify_risk_band(self, score: float) -> str:
        if score < 30:
            return "Conservative"
        elif score < 55:
            return "Moderate"
        elif score < 80:
            return "Growth"
        else:
            return "Aggressive"

    def recommend_portfolio(self, profile: dict) -> dict:
        score = self.compute_risk_score(profile)
        band = self.classify_risk_band(score)
        allocation = PORTFOLIOS[band]

        rationale = (
            f"Based on a risk-tolerance score of {score}/100 (derived from age, "
            f"investment horizon, savings capacity, self-reported loss tolerance and "
            f"experience), the recommended risk band is '{band}'."
        )

        return {
            "risk_score": score,
            "risk_band": band,
            "recommended_allocation": allocation,
            "rationale": rationale,
        }


if __name__ == "__main__":
    advisor = RoboAdvisor()
    sample_profile = {
        "age": 27,
        "investment_horizon_years": 15,
        "monthly_surplus_pct": 30,
        "loss_tolerance": 0.7,
        "investment_experience": 0.4,
    }
    print(advisor.recommend_portfolio(sample_profile))
