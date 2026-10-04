import pandas as pd

from strategies import ema_crossover_strategy


data = pd.read_csv(
    "../data/eurusd_h1.csv",
    index_col=0,
    parse_dates=True
)

result = ema_crossover_strategy(data)

print(result[[
    "Close",
    "EMA_fast",
    "EMA_slow",
    "Signal"
]].tail(20))