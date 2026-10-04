import pandas as pd
from backtest import run_strategy


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


result = run_strategy(
    config,
    data
)


print("\n===== CONFIG TEST =====")

print("Strategy:", result["strategy"])
print("Fast EMA:", result["fast_ema"])
print("Slow EMA:", result["slow_ema"])
print("Return:", round(result["return_percent"], 2), "%")
print("Trades:", result["trades"])
print("Win Rate:", round(result["win_rate"], 2), "%")
print("Profit Factor:", round(result["profit_factor"], 2))
print(
    "Maximum Drawdown:",
    round(result["max_drawdown_percent"], 2),
    "%"
)
print(
    "Sharpe Ratio:",
    round(result["sharpe_ratio"], 2)
)