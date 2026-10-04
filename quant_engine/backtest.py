import pandas as pd
import numpy as np

from strategies import (
    ema_crossover_strategy,
    rsi_mean_reversion_strategy,
    supertrend_strategy
)


# =========================================================
# Strategy Registry
# =========================================================

STRATEGY_REGISTRY = {
    "EMA_Crossover": ema_crossover_strategy,
    "RSI_Mean_Reversion": rsi_mean_reversion_strategy,
    "Supertrend": supertrend_strategy
}


def get_strategy_function(strategy_name):
    """
    Return the Python implementation associated with
    the requested strategy.
    """

    strategy_function = STRATEGY_REGISTRY.get(
        strategy_name
    )

    if strategy_function is None:

        raise ValueError(
            "Unknown strategy: " +
            str(strategy_name)
        )

    return strategy_function


# =========================================================
# Utility Functions
# =========================================================

def get_timestamp(data, index):
    """
    Safely obtain a timezone-consistent timestamp
    from the Datetime column.
    """

    if "Datetime" not in data.columns:
        return None

    value = data["Datetime"].iloc[index]

    try:
        timestamp = pd.to_datetime(
            value,
            utc=True
        )

        return timestamp

    except Exception:
        return value


def calculate_trade_drawdown(
    balance,
    running_peak
):
    """
    Calculate drawdown percentage from the current
    balance and the running peak.
    """

    if running_peak == 0:
        return 0.0

    return (
        (balance - running_peak)
        / running_peak
        * 100
    )


def calculate_rolling_sharpe(
    returns,
    window=20
):
    """
    Calculate a rolling Sharpe ratio.

    This is used for the research report chart.
    It is separate from the final overall Sharpe ratio.
    """

    values = []

    returns = list(returns)

    for i in range(len(returns)):

        start = max(
            0,
            i - window + 1
        )

        window_returns = returns[
            start:i + 1
        ]

        if len(window_returns) < 2:

            values.append(0.0)

            continue

        series = pd.Series(
            window_returns,
            dtype=float
        )

        std = series.std()

        if std == 0 or pd.isna(std):

            values.append(0.0)

            continue

        sharpe = (
            series.mean()
            / std
            * np.sqrt(252)
        )

        values.append(
            float(sharpe)
        )

    return values


# =========================================================
# Main Backtest
# =========================================================

def run_backtest(
    data,
    strategy_config,
    initial_capital=10000
):
    """
    Run a complete deterministic strategy backtest.

    The LLM supplies strategy_config.

    Python performs:
        - strategy calculation
        - signal generation
        - position management
        - spread
        - stop loss
        - take profit
        - trade accounting
        - performance metrics
        - detailed trade analysis
    """

    # -----------------------------------------------------
    # Validate input data
    # -----------------------------------------------------

    if not isinstance(data, pd.DataFrame):

        raise ValueError(
            "Backtest data must be a pandas DataFrame."
        )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                "Missing required data column: " +
                column
            )

    data = data.copy()
    # -----------------------------------------------------
    # Preserve market timestamps before resetting the index.
    #
    # yfinance commonly provides the timestamp as the
    # DataFrame index rather than a "Datetime" column.
    # -----------------------------------------------------
    if "Datetime" not in data.columns:
        try:
            if isinstance(
                data.index,
                pd.DatetimeIndex
            ):
                data["Datetime"] = (
                     pd.to_datetime(
                        data.index
                    )
                )
            else:
                converted_index = pd.to_datetime(
                    data.index,
                    errors="coerce"
                )
                if not converted_index.isna().all():
                    data["Datetime"] = (
                        converted_index
                    )
        except Exception:
            pass
    # -----------------------------------------------------
    # Now reset the index while keeping the timestamp
    # column that we just preserved.
    # -----------------------------------------------------                                
    data = data.reset_index(
        drop=True
    )    
    if len(data) == 0:

        raise ValueError(
            "Backtest received empty market data."
        )

    # -----------------------------------------------------
    # Extract strategy configuration
    # -----------------------------------------------------

    strategy = strategy_config.get(
        "strategy",
        "EMA_Crossover"
    )

    parameters = strategy_config.get(
        "parameters",
        {}
    )

    risk = strategy_config.get(
        "risk",
        {}
    )

    capital_config = strategy_config.get(
        "capital",
        {}
    )

    starting_capital = float(
        capital_config.get(
            "starting_capital",
            initial_capital
        )
    )

    if starting_capital <= 0:

        raise ValueError(
            "starting_capital must be greater than zero."
        )

    # -----------------------------------------------------
    # Execute strategy
    # -----------------------------------------------------

    strategy_function = get_strategy_function(
        strategy
    )

    if strategy == "EMA_Crossover":

        fast_period = int(
            parameters.get(
                "fast_ema",
                20
            )
        )

        slow_period = int(
            parameters.get(
                "slow_ema",
                50
            )
        )

        if fast_period <= 0:

            raise ValueError(
                "fast_ema must be greater than zero."
            )

        if slow_period <= fast_period:

            raise ValueError(
                "slow_ema must be greater than fast_ema."
            )

        data = strategy_function(
            data.copy(),
            fast_period,
            slow_period
        )

    elif strategy == "RSI_Mean_Reversion":

        rsi_period = int(
            parameters.get(
                "rsi_period",
                14
            )
        )

        oversold = float(
            parameters.get(
                "oversold",
                30
            )
        )

        overbought = float(
            parameters.get(
                "overbought",
                70
            )
        )

        if rsi_period <= 0:

            raise ValueError(
                "rsi_period must be greater than zero."
            )

        if oversold >= overbought:

            raise ValueError(
                "oversold must be less than overbought."
            )

        data = strategy_function(
            data.copy(),
            rsi_period,
            oversold,
            overbought
        )

    elif strategy == "Supertrend":

        atr_period = int(
            parameters.get(
                "atr_period",
                10
            )
        )

        multiplier = float(
            parameters.get(
                "multiplier",
                3.0
            )
        )

        if atr_period <= 0:

            raise ValueError(
                "atr_period must be greater than zero."
            )

        if multiplier <= 0:

            raise ValueError(
                "multiplier must be greater than zero."
            )

        data = strategy_function(
            data.copy(),
            atr_period,
            multiplier
        )

    else:

        raise ValueError(
            "Strategy is registered but its parameter "
            "handling has not been implemented: " +
            str(strategy)
        )

    if "Signal" not in data.columns:

        raise ValueError(
            "Strategy did not produce a Signal column."
        )

    # -----------------------------------------------------
    # Position sizing
    # -----------------------------------------------------

    lot_size = risk.get(
        "lot_size",
        None
    )

    if lot_size is None:

        position_size = 10000.0

    else:

        position_size = float(
            lot_size
        )

        if position_size <= 0:

            raise ValueError(
                "lot_size must be greater than zero."
            )

    # -----------------------------------------------------
    # Risk parameters
    #
    # SL/TP are positive price distances.
    #
    # Example EUR/USD:
    # 0.0010 = approximately 10 pips
    # 0.0020 = approximately 20 pips
    # -----------------------------------------------------

    stop_loss = risk.get(
        "stop_loss",
        None
    )

    take_profit = risk.get(
        "take_profit",
        None
    )

    if stop_loss is not None:

        stop_loss = float(
            stop_loss
        )

        if stop_loss <= 0:

            raise ValueError(
                "stop_loss must be positive or null."
            )

    if take_profit is not None:

        take_profit = float(
            take_profit
        )

        if take_profit <= 0:

            raise ValueError(
                "take_profit must be positive or null."
            )

    # -----------------------------------------------------
    # Trading state
    # -----------------------------------------------------

    capital = starting_capital

    position = 0

    entry_price = None

    entry_index = None

    entry_time = None

    trade_number = 0

    spread = 0.0001

    # -----------------------------------------------------
    # Research data structures
    # -----------------------------------------------------

    trades_detail = []

    equity_curve = []

    trade_balance_curve = []

    trade_drawdown_curve = []

    trade_returns = []

    running_peak = starting_capital

    # =====================================================
    # Helper: Close Position
    # =====================================================

    def close_position(
        exit_price,
        exit_index,
        exit_reason
    ):

        nonlocal capital
        nonlocal position
        nonlocal entry_price
        nonlocal entry_index
        nonlocal entry_time
        nonlocal trade_number
        nonlocal running_peak

        if position == 0:
            return

        exit_time = get_timestamp(
            data,
            exit_index
        )

        # -------------------------------------------------
        # Calculate profit
        # -------------------------------------------------

        if position == 1:

            profit = (
                exit_price
                - entry_price
            ) * position_size

            direction = "LONG"

        else:

            profit = (
                entry_price
                - exit_price
            ) * position_size

            direction = "SHORT"

        capital += profit

        trade_number += 1

        # -------------------------------------------------
        # Update running peak
        # -------------------------------------------------

        running_peak = max(
            running_peak,
            capital
        )

        drawdown = calculate_trade_drawdown(
            capital,
            running_peak
        )

        # -------------------------------------------------
        # Trade return relative to
        # previous balance
        # -------------------------------------------------

        previous_balance = (
            capital - profit
        )

        if previous_balance != 0:

            trade_return = (
                profit
                / previous_balance
            )

        else:

            trade_return = 0.0

        trade_returns.append(
            float(trade_return)
        )

        # -------------------------------------------------
        # Weekday
        # -------------------------------------------------

        weekday = None

        if exit_time is not None:

            try:

                weekday = exit_time.day_name()

            except Exception:

                weekday = None

        # -------------------------------------------------
        # Store detailed trade
        # -------------------------------------------------

        trade_record = {

            "trade_number":
                trade_number,

            "entry_time":
                entry_time,

            "exit_time":
                exit_time,

            "weekday":
                weekday,

            "position":
                direction,

            "entry_price":
                float(entry_price),

            "exit_price":
                float(exit_price),

            "profit":
                float(profit),

            "balance_after_trade":
                float(capital),

            "drawdown_percent":
                float(drawdown),

            "exit_reason":
                exit_reason
        }

        trades_detail.append(
            trade_record
        )

        # -------------------------------------------------
        # Curves used by the PDF
        # -------------------------------------------------

        trade_balance_curve.append({

            "trade_number":
                trade_number,

            "timestamp":
                exit_time,

            "balance":
                float(capital)
        })

        trade_drawdown_curve.append({

            "trade_number":
                trade_number,

            "timestamp":
                exit_time,

            "drawdown_percent":
                float(drawdown)
        })

        # -------------------------------------------------
        # Reset position
        # -------------------------------------------------

        position = 0

        entry_price = None

        entry_index = None

        entry_time = None

    # =====================================================
    # Main Candle Loop
    # =====================================================

    for i in range(len(data)):

        close_price = float(
            data["Close"].iloc[i]
        )

        high_price = float(
            data["High"].iloc[i]
        )

        low_price = float(
            data["Low"].iloc[i]
        )

        signal = data["Signal"].iloc[i]

        # -------------------------------------------------
        # Current unrealized P&L
        # -------------------------------------------------

        if position == 1:

            unrealized_pnl = (
                close_price
                - entry_price
            ) * position_size

        elif position == -1:

            unrealized_pnl = (
                entry_price
                - close_price
            ) * position_size

        else:

            unrealized_pnl = 0.0

        current_equity = (
            capital
            + unrealized_pnl
        )

        equity_curve.append(
            float(current_equity)
        )

        # -------------------------------------------------
        # Check Stop Loss / Take Profit
        # -------------------------------------------------

        if position == 1:

            stop_price = None

            target_price = None

            if stop_loss is not None:

                stop_price = (
                    entry_price
                    - stop_loss
                )

            if take_profit is not None:

                target_price = (
                    entry_price
                    + take_profit
                )

            stop_hit = (
                stop_price is not None
                and low_price <= stop_price
            )

            target_hit = (
                target_price is not None
                and high_price >= target_price
            )

            # If both are hit in the same candle,
            # assume stop loss happened first.

            if stop_hit:

                close_position(
                    stop_price,
                    i,
                    "stop_loss"
                )

            elif target_hit:

                close_position(
                    target_price,
                    i,
                    "take_profit"
                )

        elif position == -1:

            stop_price = None

            target_price = None

            if stop_loss is not None:

                stop_price = (
                    entry_price
                    + stop_loss
                )

            if take_profit is not None:

                target_price = (
                    entry_price
                    - take_profit
                )

            stop_hit = (
                stop_price is not None
                and high_price >= stop_price
            )

            target_hit = (
                target_price is not None
                and low_price <= target_price
            )

            # Conservative assumption:
            # stop loss takes priority.

            if stop_hit:

                close_position(
                    stop_price,
                    i,
                    "stop_loss"
                )

            elif target_hit:

                close_position(
                    target_price,
                    i,
                    "take_profit"
                )

        # -------------------------------------------------
        # Process strategy signal
        # -------------------------------------------------

        if signal == 1 and position == 0:

            position = 1

            entry_price = (
                close_price
                + spread
            )

            entry_index = i

            entry_time = get_timestamp(
                data,
                i
            )

        elif signal == -1 and position == 0:

            position = -1

            entry_price = (
                close_price
                - spread
            )

            entry_index = i

            entry_time = get_timestamp(
                data,
                i
            )

        elif signal == -1 and position == 1:

            close_position(
                close_price,
                i,
                "signal"
            )

            # Open short after closing long.

            position = -1

            entry_price = (
                close_price
                - spread
            )

            entry_index = i

            entry_time = get_timestamp(
                data,
                i
            )

        elif signal == 1 and position == -1:

            close_position(
                close_price,
                i,
                "signal"
            )

            # Open long after closing short.

            position = 1

            entry_price = (
                close_price
                + spread
            )

            entry_index = i

            entry_time = get_timestamp(
                data,
                i
            )

    # =====================================================
    # Close Remaining Position
    # =====================================================

    final_index = len(data) - 1

    final_price = float(
        data["Close"].iloc[-1]
    )

    if position == 1:

        close_position(
            final_price,
            final_index,
            "end_of_data"
        )

    elif position == -1:

        close_position(
            final_price,
            final_index,
            "end_of_data"
        )

    # =====================================================
    # Final Equity
    # =====================================================

    equity_curve.append(
        float(capital)
    )

    # =====================================================
    # Performance Metrics
    # =====================================================

    equity = pd.Series(
        equity_curve,
        dtype=float
    )

    total_return = (
        (
            capital
            - starting_capital
        )
        / starting_capital
        * 100
    )

    running_max = equity.cummax()

    drawdown = (
        (
            equity
            - running_max
        )
        / running_max
    )

    max_drawdown = (
        drawdown.min()
        * 100
    )

    # -----------------------------------------------------
    # Overall Sharpe Ratio
    # -----------------------------------------------------

    returns = (
        equity
        .pct_change()
        .dropna()
    )

    if (
        len(returns) > 1
        and returns.std() != 0
    ):

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(24 * 252)
        )

    else:

        sharpe = 0.0

    # =====================================================
    # Trade Statistics
    # =====================================================

    profits = [
        trade["profit"]
        for trade in trades_detail
    ]

    if len(profits) > 0:

        winning_trades = [
            profit
            for profit in profits
            if profit > 0
        ]

        losing_trades = [
            profit
            for profit in profits
            if profit < 0
        ]

        win_rate = (
            len(winning_trades)
            / len(profits)
            * 100
        )

        total_profit = sum(
            winning_trades
        )

        total_loss = abs(
            sum(losing_trades)
        )

        if total_loss > 0:

            profit_factor = (
                total_profit
                / total_loss
            )

        else:

            profit_factor = 0.0

    else:

        win_rate = 0.0

        profit_factor = 0.0

    # =====================================================
    # Rolling Sharpe
    # =====================================================

    rolling_sharpe_values = (
        calculate_rolling_sharpe(
            trade_returns,
            window=20
        )
    )

    rolling_sharpe = []

    for i, value in enumerate(
        rolling_sharpe_values
    ):

        timestamp = None

        if i < len(trades_detail):

            timestamp = (
                trades_detail[i]
                .get("exit_time")
            )

        rolling_sharpe.append({

            "trade_number":
                i + 1,

            "timestamp":
                timestamp,

            "sharpe_ratio":
                float(value)
        })

    # =====================================================
    # Weekday Analysis
    # =====================================================

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_analysis = []

    for weekday in weekday_order:

        weekday_trades = [
            trade
            for trade in trades_detail
            if trade["weekday"] == weekday
        ]

        if len(weekday_trades) == 0:

            weekday_analysis.append({

                "weekday":
                    weekday,

                "trades":
                    0,

                "profit":
                    0.0,

                "average_profit":
                    0.0,

                "win_rate":
                    0.0
            })

            continue

        weekday_profits = [
            trade["profit"]
            for trade in weekday_trades
        ]

        winning = [
            profit
            for profit in weekday_profits
            if profit > 0
        ]

        weekday_analysis.append({

            "weekday":
                weekday,

            "trades":
                len(weekday_trades),

            "profit":
                float(
                    sum(weekday_profits)
                ),

            "average_profit":
                float(
                    np.mean(
                        weekday_profits
                    )
                ),

            "win_rate":
                float(
                    len(winning)
                    / len(weekday_profits)
                    * 100
                )
        })

    # =====================================================
    # Exit Reason Statistics
    # =====================================================

    stop_loss_exits = sum(
        1
        for trade in trades_detail
        if trade["exit_reason"]
        == "stop_loss"
    )

    take_profit_exits = sum(
        1
        for trade in trades_detail
        if trade["exit_reason"]
        == "take_profit"
    )

    signal_exits = sum(
        1
        for trade in trades_detail
        if trade["exit_reason"]
        == "signal"
    )

    end_of_data_exits = sum(
        1
        for trade in trades_detail
        if trade["exit_reason"]
        == "end_of_data"
    )

    # =====================================================
    # Trade Balance / Drawdown Curves
    # =====================================================

    if len(trade_balance_curve) == 0:

        trade_balance_curve = [{

            "trade_number":
                0,

            "timestamp":
                None,

            "balance":
                float(starting_capital)
        }]

        trade_drawdown_curve = [{

            "trade_number":
                0,

            "timestamp":
                None,

            "drawdown_percent":
                0.0
        }]

    # =====================================================
    # Deterministic Heat Evaluation
    # =====================================================
    #
    # Fixed baseline:
    #
    #   Return        > 0%
    #   Profit Factor > 1
    #   Sharpe        > 0
    #   Drawdown      > -10%
    #
    # Heat score:
    #
    #   0 = COLD
    #   1 = COOL
    #   2 = WARM
    #   3 = HOT
    #   4 = VERY HOT
    #
    # Python calculates this deterministically.
    # The LLM only receives the result and reasons about it.
    # =====================================================

    heat_checks = {

        "positive_return":
            bool(
                total_return > 0
            ),

        "profit_factor_above_1":
            bool(
                profit_factor > 1
            ),

        "positive_sharpe":
            bool(
                sharpe > 0
            ),

        "drawdown_better_than_10_percent":
            bool(
                max_drawdown > -10
            )
    }

    heat_score = int(
        sum(
            1
            for passed in heat_checks.values()
            if passed
        )
    )

    heat_labels = {

        0: "COLD",
        1: "COOL",
        2: "WARM",
        3: "HOT",
        4: "VERY HOT"
    }

    heat_label = heat_labels[
        heat_score
    ]

    # =====================================================
    # Return Result
    # =====================================================

    result = {

        "strategy":
            strategy,

        "strategy_config":
            strategy_config,

        "initial_capital":
            float(starting_capital),

        "final_capital":
            float(capital),

        "return_percent":
            float(total_return),

        "trades":
            len(trades_detail),

        "win_rate":
            float(win_rate),

        "profit_factor":
            float(profit_factor),

        "max_drawdown_percent":
            float(max_drawdown),

        "sharpe_ratio":
            float(sharpe),

        # -------------------------------------------------
        # Deterministic heat evaluation
        # -------------------------------------------------

        "heat_score":
            heat_score,

        "heat_label":
            heat_label,

        "heat_checks":
            heat_checks,

        "baseline_thresholds": {

            "return_percent":
                "> 0",

            "profit_factor":
                "> 1",

            "sharpe_ratio":
                "> 0",

            "max_drawdown_percent":
                "> -10"
        },

        # -------------------------------------------------
        # Existing equity curve
        # -------------------------------------------------

        "equity_curve":
            [
                float(value)
                for value in equity_curve
            ],

        # -------------------------------------------------
        # Detailed trade information
        # -------------------------------------------------

        "trades_detail":
            trades_detail,

        # -------------------------------------------------
        # Report chart data
        # -------------------------------------------------

        "trade_balance_curve":
            trade_balance_curve,

        "trade_drawdown_curve":
            trade_drawdown_curve,

        "rolling_sharpe":
            rolling_sharpe,

        "weekday_analysis":
            weekday_analysis,

        # -------------------------------------------------
        # Exit analysis
        # -------------------------------------------------

        "stop_loss_exits":
            stop_loss_exits,

        "take_profit_exits":
            take_profit_exits,

        "signal_exits":
            signal_exits,

        "end_of_data_exits":
            end_of_data_exits,

        # -------------------------------------------------
        # Risk configuration
        # -------------------------------------------------

        "position_size":
            float(position_size),

        "stop_loss":
            stop_loss,

        "take_profit":
            take_profit,

        "lot_size":
            lot_size
    }

    # =====================================================
    # Strategy-specific compatibility fields
    # =====================================================

    if strategy == "EMA_Crossover":

        result["fast_ema"] = int(
            parameters.get(
                "fast_ema",
                20
            )
        )

        result["slow_ema"] = int(
            parameters.get(
                "slow_ema",
                50
            )
        )

    elif strategy == "RSI_Mean_Reversion":

        result["rsi_period"] = int(
            parameters.get(
                "rsi_period",
                14
            )
        )

        result["oversold"] = float(
            parameters.get(
                "oversold",
                30
            )
        )

        result["overbought"] = float(
            parameters.get(
                "overbought",
                70
            )
        )

    elif strategy == "Supertrend":

        result["atr_period"] = int(
            parameters.get(
                "atr_period",
                10
            )
        )

        result["multiplier"] = float(
            parameters.get(
                "multiplier",
                3.0
            )
        )

    return result


# =========================================================
# Legacy / API Entry Point
# =========================================================

def run_strategy(
    config,
    data
):
    """
    Compatibility wrapper used by the FastAPI layer.
    """

    strategy_config = config.get(
        "strategy_config"
    )

    if not isinstance(
        strategy_config,
        dict
    ):

        strategy = config.get(
            "strategy",
            "EMA_Crossover"
        )

        if strategy == "EMA_Crossover":

            parameters = {

                "fast_ema":
                    int(
                        config.get(
                            "fast_ema",
                            20
                        )
                    ),

                "slow_ema":
                    int(
                        config.get(
                            "slow_ema",
                            50
                        )
                    )
            }

        elif strategy == "RSI_Mean_Reversion":

            parameters = {

                "rsi_period":
                    int(
                        config.get(
                            "rsi_period",
                            14
                        )
                    ),

                "oversold":
                    float(
                        config.get(
                            "oversold",
                            30
                        )
                    ),

                "overbought":
                    float(
                        config.get(
                            "overbought",
                            70
                        )
                    )
            }

        elif strategy == "Supertrend":

            parameters = {

                "atr_period":
                    int(
                        config.get(
                            "atr_period",
                            10
                        )
                    ),

                "multiplier":
                    float(
                        config.get(
                            "multiplier",
                            3.0
                        )
                    )
            }

        else:

            parameters = {}

        strategy_config = {

            "strategy":
                strategy,

            "parameters":
                parameters,

            "risk": {

                "stop_loss":
                    config.get(
                        "stop_loss",
                        None
                    ),

                "take_profit":
                    config.get(
                        "take_profit",
                        None
                    ),

                "lot_size":
                    config.get(
                        "lot_size",
                        None
                    )
            },

            "capital": {

                "starting_capital":
                    float(
                        config.get(
                            "starting_capital",
                            10000
                        )
                    )
            }
        }

    return run_backtest(

        data=data,

        strategy_config=strategy_config,

        initial_capital=float(
            config.get(
                "starting_capital",
                10000
            )
        )
    )