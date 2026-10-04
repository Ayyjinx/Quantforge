import pandas as pd
from backtest import run_strategy
import json

def evaluate_strategy(config, data):

    result = run_strategy(
        config,
        data
    )

    clean_result = {
        "strategy": str(result["strategy"]),
        "fast_ema": int(result["fast_ema"]),
        "slow_ema": int(result["slow_ema"]),
        "return_percent": float(round
            (result["return_percent"], 2)
        ),
        "trades": int(result["trades"]),
        "win_rate": float(
            round(result["win_rate"], 2)
        ),
        "profit_factor": float(
            round(result["profit_factor"], 2)
        ),
        "max_drawdown_percent": float(
            round(result["max_drawdown_percent"], 2)
        ),
        "sharpe_ratio": float(
            round(result["sharpe_ratio"], 2)
        )
    }

    return clean_result


if __name__ == "__main__":

    data = pd.read_csv(
        "../data/eurusd_h1.csv",
        index_col=0,
        parse_dates=True
    )

    config = {
        "strategy": "EMA_Crossover",
        "fast_ema": 10,
        "slow_ema": 30
    }

    result = evaluate_strategy(
        config,
        data
    )

    print("\n===== AGENT RESULT =====")

    print(json.dumps(result, indent=2))