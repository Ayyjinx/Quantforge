import pandas as pd
from backtest import run_backtest


data = pd.read_csv(
    "../data/eurusd_h1.csv",
    index_col=0,
    parse_dates=True
)


result = run_backtest(
    data,
    fast_period=20,
    slow_period=50
)


print("\n===== BACKTEST RESULT =====")

print(
    "Strategy:",
    result["strategy"]
)

print(
    "Fast EMA:",
    result["fast_ema"]
)

print(
    "Slow EMA:",
    result["slow_ema"]
)

print(
    "Initial Capital:",
    round(result["initial_capital"], 2)
)

print(
    "Final Capital:",
    round(result["final_capital"], 2)
)

print(
    "Return:",
    round(result["return_percent"], 2),
    "%"
)

print(
    "Number of Trades:",
    result["trades"]
)

print(
    "Win Rate:",
    round(result["win_rate"], 2),
    "%"
)

print(
    "Profit Factor:",
    round(result["profit_factor"], 2)
)

print(
    "Maximum Drawdown:",
    round(result["max_drawdown_percent"], 2),
    "%"
)

print(
    "Sharpe Ratio:",
    round(result["sharpe_ratio"], 2)
)