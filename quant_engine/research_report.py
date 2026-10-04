import json


def generate_report(
    research_objective,
    history
):

    if len(history) == 0:
        return {
            "research_objective": research_objective,
            "experiments": [],
            "best_strategy": None
        }

    # Find best experiment using research score
    best = history[0]

    for experiment in history:

        if (
            experiment["research_score"]
            > best["research_score"]
        ):
            best = experiment

    report = {
        "research_objective": research_objective,
        "experiments_completed": len(history),
        "experiments": history,
        "best_strategy": best
    }

    return report


if __name__ == "__main__":

    # Example history from our previous run
    history = [
        {
            "strategy": "EMA_Crossover",
            "fast_ema": 12,
            "slow_ema": 30,
            "return_percent": 0.64,
            "trades": 398,
            "win_rate": 28.39,
            "profit_factor": 1.01,
            "max_drawdown_percent": -8.21,
            "sharpe_ratio": 0.08,
            "research_score": -1.93
        },
        {
            "strategy": "EMA_Crossover",
            "fast_ema": 15,
            "slow_ema": 50,
            "return_percent": -1.34,
            "trades": 269,
            "win_rate": 29.0,
            "profit_factor": 0.98,
            "max_drawdown_percent": -9.09,
            "sharpe_ratio": -0.04,
            "research_score": -7.83
        }
    ]

    objective = """
    Find a simple EMA crossover strategy for EUR/USD
    on the H1 timeframe.
    """

    report = generate_report(
        objective,
        history
    )

    print("\n===== RESEARCH REPORT =====")

    print(
        json.dumps(
            report,
            indent=2
        )
    )