import pandas as pd

from indicators import calculate_ema, calculate_rsi, calculate_atr


# Load the data
data = pd.read_csv(
    "../data/eurusd_h1.csv",
    index_col=0,
    parse_dates=True
)

# Calculate indicators
data["EMA_20"] = calculate_ema(data, 20)
data["RSI_14"] = calculate_rsi(data, 14)
data["ATR_14"] = calculate_atr(data, 14)

# Show the results
print(data[["Close", "EMA_20", "RSI_14", "ATR_14"]].tail(10))