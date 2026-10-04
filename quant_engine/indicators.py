import pandas as pd


def calculate_ema(data, period):
    return data["Close"].ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period=14):
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(period).mean()
    average_loss = loss.rolling(period).mean()

    rs = average_gain / average_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(data, period=14):
    high_low = data["High"] - data["Low"]

    high_close = abs(data["High"] - data["Close"].shift())
    low_close = abs(data["Low"] - data["Close"].shift())

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(period).mean()

    return atr