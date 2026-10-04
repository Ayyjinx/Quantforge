def calculate_research_score(result):

    score = 0

    # Return contribution
    score += result["return_percent"] 

    # Sharpe contribution
    score += result["sharpe_ratio"] * 10

    # Profit factor contribution
    score += (
        result["profit_factor"] - 1
    ) * 20

    # Drawdown penalty
    score += abs(result["max_drawdown_percent"]) * 0.5

    return round(score, 2)


def validate_strategy(result):

    checks = {}

    # Strategy should make money
    checks["positive_return"] = (
        result["return_percent"] > 0
    )

    # Profit factor should be above 1
    checks["profit_factor"] = (
        result["profit_factor"] > 1
    )

    # Sharpe should be positive
    checks["sharpe_ratio"] = (
        result["sharpe_ratio"] > 0
    )

    # Require enough trades
    checks["minimum_trades"] = (
        result["trades"] >= 100
    )

    # Maximum acceptable drawdown
    checks["maximum_drawdown"] = (
        result["max_drawdown_percent"] >= -10
    )

    passed = all(checks.values())

    score = calculate_research_score(result)

    return {
        "passed": passed,
        "score": score,
        "checks": checks
    }


if __name__ == "__main__":

    test_result = {
        "return_percent": 0.67,
        "trades": 171,
        "profit_factor": 1.02,
        "max_drawdown_percent": -8.04,
        "sharpe_ratio": 0.08
    }

    validation = validate_strategy(
        test_result
    )

    print("\n===== VALIDATION =====")

    print("Passed:", validation["passed"])

    print("Score:", validation["score"])

    print("\nChecks:")

    for name, passed in validation["checks"].items():

        print(
            name + ":",
            "PASS" if passed else "FAIL"
        )