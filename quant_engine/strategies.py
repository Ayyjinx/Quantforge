import pandas as pd
import numpy as np
from indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr
)


# ---------------------------------------------------------
# EMA Crossover Strategy
# ---------------------------------------------------------

def ema_crossover_strategy(
    data,
    fast_period=20,
    slow_period=50
):

    data = data.copy()

    data["EMA_fast"] = calculate_ema(
        data,
        fast_period
    )

    data["EMA_slow"] = calculate_ema(
        data,
        slow_period
    )

    data["Signal"] = 0

    data.loc[
        (data["EMA_fast"] > data["EMA_slow"]) &
        (
            data["EMA_fast"].shift(1)
            <=
            data["EMA_slow"].shift(1)
        ),
        "Signal"
    ] = 1

    data.loc[
        (data["EMA_fast"] < data["EMA_slow"]) &
        (
            data["EMA_fast"].shift(1)
            >=
            data["EMA_slow"].shift(1)
        ),
        "Signal"
    ] = -1

    return data


# ---------------------------------------------------------
# RSI Mean Reversion Strategy
# ---------------------------------------------------------

def rsi_mean_reversion_strategy(
    data,
    rsi_period=14,
    oversold=30,
    overbought=70
):

    data = data.copy()

    data["RSI"] = calculate_rsi(
        data,
        rsi_period
    )

    data["Signal"] = 0

    data.loc[
        (data["RSI"] > oversold) &
        (
            data["RSI"].shift(1)
            <=
            oversold
        ),
        "Signal"
    ] = 1

    data.loc[
        (data["RSI"] < overbought) &
        (
            data["RSI"].shift(1)
            >=
            overbought
        ),
        "Signal"
    ] = -1

    return data


# ---------------------------------------------------------
# Supertrend Strategy
# ---------------------------------------------------------

def supertrend_strategy(
    data,
    atr_period=10,
    multiplier=3.0
):
    """
    Supertrend trend-following strategy.

    Parameters
    ----------
    atr_period : int
        ATR lookback period.

    multiplier : float
        ATR multiplier used to calculate
        Supertrend bands.

    Signals
    -------
    1
        Bullish Supertrend flip.

    -1
        Bearish Supertrend flip.

    0
        No direction change.
    """

    data = data.copy()

    # -----------------------------------------------------
    # Validate parameters
    # -----------------------------------------------------

    if atr_period <= 0:

        raise ValueError(
            "atr_period must be greater than zero."
        )

    if multiplier <= 0:

        raise ValueError(
            "multiplier must be greater than zero."
        )

    if len(data) == 0:

        raise ValueError(
            "Supertrend received empty market data."
        )

    # -----------------------------------------------------
    # Calculate ATR
    # -----------------------------------------------------

    atr = calculate_atr(
        data,
        atr_period
    )

    # -----------------------------------------------------
    # Basic bands
    # -----------------------------------------------------

    hl2 = (
        data["High"]
        +
        data["Low"]
    ) / 2.0

    basic_upper = (
        hl2
        +
        multiplier * atr
    )

    basic_lower = (
        hl2
        -
        multiplier * atr
    )

    # -----------------------------------------------------
    # Final bands
    # -----------------------------------------------------

    final_upper = pd.Series(
        index=data.index,
        dtype=float
    )

    final_lower = pd.Series(
        index=data.index,
        dtype=float
    )

    # -----------------------------------------------------
    # Direction
    #
    #  1 = bullish
    # -1 = bearish
    # -----------------------------------------------------

    direction = pd.Series(
        index=data.index,
        dtype=int
    )

    supertrend = pd.Series(
        index=data.index,
        dtype=float
    )

    # -----------------------------------------------------
    # Signal
    # -----------------------------------------------------

    signal = pd.Series(
        0,
        index=data.index,
        dtype=int
    )

    # -----------------------------------------------------
    # Find first valid ATR row
    # -----------------------------------------------------

    first_valid = atr.first_valid_index()

    if first_valid is None:

        data["Supertrend"] = np.nan

        data["Supertrend_direction"] = 0

        data["Signal"] = 0

        return data

    first_position = data.index.get_loc(
        first_valid
    )

    # Before ATR becomes available, keep
    # the strategy neutral.

    for i in range(
        0,
        first_position
    ):

        final_upper.iloc[i] = np.nan

        final_lower.iloc[i] = np.nan

        direction.iloc[i] = 0

        supertrend.iloc[i] = np.nan

    # -----------------------------------------------------
    # Initialize first valid row
    # -----------------------------------------------------

    final_upper.iloc[first_position] = (
        basic_upper.iloc[first_position]
    )

    final_lower.iloc[first_position] = (
        basic_lower.iloc[first_position]
    )

    direction.iloc[first_position] = 1

    supertrend.iloc[first_position] = (
        final_lower.iloc[first_position]
    )

    # -----------------------------------------------------
    # Main Supertrend calculation
    # -----------------------------------------------------

    for i in range(
        first_position + 1,
        len(data)
    ):

        current_close = float(
            data["Close"].iloc[i]
        )

        previous_close = float(
            data["Close"].iloc[i - 1]
        )

        previous_upper = (
            final_upper.iloc[i - 1]
        )

        previous_lower = (
            final_lower.iloc[i - 1]
        )

        previous_direction = (
            direction.iloc[i - 1]
        )

        # -------------------------------------------------
        # Final upper band
        # -------------------------------------------------

        if (
            basic_upper.iloc[i]
            < previous_upper
            or
            previous_close
            > previous_upper
        ):

            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )

        else:

            final_upper.iloc[i] = (
                previous_upper
            )

        # -------------------------------------------------
        # Final lower band
        # -------------------------------------------------

        if (
            basic_lower.iloc[i]
            > previous_lower
            or
            previous_close
            < previous_lower
        ):

            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )

        else:

            final_lower.iloc[i] = (
                previous_lower
            )

        # -------------------------------------------------
        # Determine current direction
        # -------------------------------------------------

        if previous_direction == 1:

            if current_close <= final_lower.iloc[i]:

                current_direction = -1

            else:

                current_direction = 1

        else:

            if current_close >= final_upper.iloc[i]:

                current_direction = 1

            else:

                current_direction = -1

        direction.iloc[i] = (
            current_direction
        )

        # -------------------------------------------------
        # Current Supertrend value
        # -------------------------------------------------

        if current_direction == 1:

            supertrend.iloc[i] = (
                final_lower.iloc[i]
            )

        else:

            supertrend.iloc[i] = (
                final_upper.iloc[i]
            )

        # -------------------------------------------------
        # Detect direction change
        # -------------------------------------------------

        if (
            previous_direction == -1
            and
            current_direction == 1
        ):

            signal.iloc[i] = 1

        elif (
            previous_direction == 1
            and
            current_direction == -1
        ):

            signal.iloc[i] = -1

    # -----------------------------------------------------
    # Store calculated Supertrend
    # -----------------------------------------------------

    data["Supertrend"] = (
        supertrend
    )

    data["Supertrend_direction"] = (
        direction
    )

    data["Signal"] = (
        signal
    )

    return data