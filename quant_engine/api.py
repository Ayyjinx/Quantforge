from fastapi import FastAPI, HTTPException
import math
import numpy as np
import pandas as pd

from backtest import run_strategy
from data_loader import load_market_data


app = FastAPI()


# ============================================================
# BASIC API
# ============================================================

@app.get("/")
def home():

    return {
        "status": "QuantForge API is running"
    }


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(value, default=0.0):

    try:

        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# WEEKDAY ANALYSIS
#
# Prefer deterministic calculation directly from
# trades_detail.
# ============================================================

def build_weekday_analysis(trades_detail):

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    grouped = {
        day: []
        for day in weekday_order
    }

    if not isinstance(
        trades_detail,
        list
    ):
        trades_detail = []

    for trade in trades_detail:

        if not isinstance(
            trade,
            dict
        ):
            continue

        # ====================================================
        # ALWAYS derive weekday from exit_time
        # ====================================================

        exit_time = trade.get(
            "exit_time"
        )

        if exit_time is None:
            continue

        try:

            exit_timestamp = pd.to_datetime(
                exit_time
            )

            if pd.isna(
                exit_timestamp
            ):
                continue

            weekday = (
                exit_timestamp.day_name()
            )

        except Exception:

            continue

        if weekday not in grouped:
            continue

        profit = safe_float(
            trade.get(
                "profit",
                0
            )
        )

        grouped[
            weekday
        ].append(
            profit
        )

    # ========================================================
    # Build final seven-day table
    # ========================================================

    result = []

    for weekday in weekday_order:

        profits = grouped[
            weekday
        ]

        trade_count = len(
            profits
        )

        if trade_count == 0:

            result.append({
                "weekday": weekday,
                "trades": 0,
                "profit": 0.0,
                "average_profit": 0.0,
                "win_rate": 0.0
            })

            continue

        total_profit = sum(
            profits
        )

        average_profit = (
            total_profit
            / trade_count
        )

        winning_trades = sum(
            1
            for profit in profits
            if profit > 0
        )

        win_rate = (
            winning_trades
            / trade_count
            * 100
        )

        result.append({
            "weekday": weekday,

            "trades":
                int(trade_count),

            "profit":
                float(total_profit),

            "average_profit":
                float(average_profit),

            "win_rate":
                float(win_rate)
        })

    return result

# ============================================================
# ROBUST ROLLING SHARPE
#
# We deliberately require a full 20-trade window.
#
# This prevents meaningless Sharpe values from the first
# 2-3 trades where standard deviation can be almost zero.
# ============================================================

def build_rolling_sharpe(
    trades_detail,
    window=20
):

    if not isinstance(
        trades_detail,
        list
    ):

        return []

    trade_returns = []

    for trade in trades_detail:

        if not isinstance(
            trade,
            dict
        ):

            continue

        profit = safe_float(
            trade.get(
                "profit",
                0
            )
        )

        previous_balance = safe_float(
            trade.get(
                "balance_after_trade",
                0
            )
        )

        # ----------------------------------------------------
        # Recover previous balance.
        #
        # balance_after_trade = previous_balance + profit
        # ----------------------------------------------------

        previous_balance = (
            previous_balance
            - profit
        )

        if previous_balance == 0:

            trade_return = 0.0

        else:

            trade_return = (
                profit
                / previous_balance
            )

        if not math.isfinite(
            trade_return
        ):

            trade_return = 0.0

        trade_returns.append(
            float(trade_return)
        )

    result = []

    for i in range(
        len(trade_returns)
    ):

        # ----------------------------------------------------
        # Before 20 completed trades there is not enough
        # information for a meaningful rolling Sharpe.
        # ----------------------------------------------------

        if i + 1 < window:

            value = None

        else:

            current_window = (
                trade_returns[
                    i - window + 1:
                    i + 1
                ]
            )

            series = pd.Series(
                current_window,
                dtype=float
            )

            mean_return = float(
                series.mean()
            )

            std_return = float(
                series.std(
                    ddof=1
                )
            )

            # ------------------------------------------------
            # Protect against zero / near-zero volatility.
            # ------------------------------------------------

            if (
                not math.isfinite(
                    std_return
                )
                or std_return < 1e-8
            ):

                value = 0.0

            else:

                value = (
                    mean_return
                    / std_return
                    * np.sqrt(252)
                )

                # ------------------------------------------------
                # Final safety check.
                # ------------------------------------------------

                if not math.isfinite(
                    value
                ):

                    value = 0.0

                # ------------------------------------------------
                # A rolling Sharpe in this research report is
                # informational. Do not allow pathological
                # numerical explosions.
                # ------------------------------------------------

                value = max(
                    min(
                        float(value),
                        10.0
                    ),
                    -10.0
                )

        trade = trades_detail[i]

        timestamp = trade.get(
            "exit_time"
        )

        result.append({
            "trade_number":
                int(
                    trade.get(
                        "trade_number",
                        i + 1
                    )
                ),

            "timestamp":
                timestamp,

            "sharpe_ratio":
                value
        })

    return result


# ============================================================
# TRADE BALANCE CURVE
# ============================================================

def build_balance_curve(
    trades_detail,
    starting_capital
):

    result = []

    if not isinstance(
        trades_detail,
        list
    ):

        trades_detail = []

    # Initial point
    result.append({
        "trade_number":
            0,

        "balance":
            float(starting_capital)
    })

    for i, trade in enumerate(
        trades_detail,
        start=1
    ):

        if not isinstance(
            trade,
            dict
        ):

            continue

        balance = safe_float(
            trade.get(
                "balance_after_trade",
                starting_capital
            ),
            starting_capital
        )

        result.append({
            "trade_number":
                int(
                    trade.get(
                        "trade_number",
                        i
                    )
                ),

            "balance":
                float(balance)
        })

    return result


# ============================================================
# TRADE DRAWDOWN CURVE
# ============================================================

def build_drawdown_curve(
    trades_detail
):

    result = []

    if not isinstance(
        trades_detail,
        list
    ):

        trades_detail = []

    result.append({
        "trade_number":
            0,

        "drawdown_percent":
            0.0
    })

    for i, trade in enumerate(
        trades_detail,
        start=1
    ):

        if not isinstance(
            trade,
            dict
        ):

            continue

        drawdown = safe_float(
            trade.get(
                "drawdown_percent",
                0
            )
        )

        result.append({
            "trade_number":
                int(
                    trade.get(
                        "trade_number",
                        i
                    )
                ),

            "drawdown_percent":
                float(drawdown)
        })

    return result


# ============================================================
# EXIT COUNTS
# ============================================================

def build_exit_counts(
    trades_detail
):

    counts = {
        "stop_loss": 0,
        "take_profit": 0,
        "signal": 0,
        "end_of_data": 0
    }

    if not isinstance(
        trades_detail,
        list
    ):

        return counts

    for trade in trades_detail:

        if not isinstance(
            trade,
            dict
        ):

            continue

        reason = str(
            trade.get(
                "exit_reason",
                ""
            )
        ).strip().lower()

        if reason in counts:

            counts[
                reason
            ] += 1

    return counts


# ============================================================
# CHART DATA
# ============================================================

def clean_chart_data(
    result,
    starting_capital
):

    trades_detail = result.get(
        "trades_detail",
        []
    )

    if not isinstance(
        trades_detail,
        list
    ):

        trades_detail = []

    # --------------------------------------------------------
    # ALWAYS derive weekday analysis from trade records.
    #
    # This avoids relying on a potentially incomplete
    # precomputed weekday_analysis field.
    # --------------------------------------------------------

    weekday_analysis = (
        build_weekday_analysis(
            trades_detail
        )
    )

    # --------------------------------------------------------
    # ALWAYS derive rolling Sharpe from trade records.
    # --------------------------------------------------------

    rolling_sharpe = (
        build_rolling_sharpe(
            trades_detail,
            window=20
        )
    )

    # --------------------------------------------------------
    # Balance curve
    # --------------------------------------------------------

    trade_balance_curve = (
        build_balance_curve(
            trades_detail,
            starting_capital
        )
    )

    # --------------------------------------------------------
    # Drawdown curve
    # --------------------------------------------------------

    trade_drawdown_curve = (
        build_drawdown_curve(
            trades_detail
        )
    )

    # --------------------------------------------------------
    # Exit counts
    # --------------------------------------------------------

    exit_counts = (
        build_exit_counts(
            trades_detail
        )
    )

    return {

        "trade_balance_curve":
            trade_balance_curve,

        "trade_drawdown_curve":
            trade_drawdown_curve,

        "rolling_sharpe":
            rolling_sharpe,

        "weekday_analysis":
            weekday_analysis,

        "stop_loss_exits":
            exit_counts[
                "stop_loss"
            ],

        "take_profit_exits":
            exit_counts[
                "take_profit"
            ],

        "signal_exits":
            exit_counts[
                "signal"
            ],

        "end_of_data_exits":
            exit_counts[
                "end_of_data"
            ]
    }


# ============================================================
# BACKTEST ENDPOINT
# ============================================================

@app.post("/backtest")
def backtest_strategy(
    config: dict
):

    try:

        strategy_config = config.get(
            "strategy_config"
        )

        if not isinstance(
            strategy_config,
            dict
        ):

            raise ValueError(
                "strategy_config must be a dictionary."
            )

        # ----------------------------------------------------
        # Strategy
        # ----------------------------------------------------

        strategy = strategy_config.get(
            "strategy",
            config.get(
                "strategy",
                "EMA_Crossover"
            )
        )

        parameters = strategy_config.get(
            "parameters",
            {}
        )

        if not isinstance(
            parameters,
            dict
        ):

            parameters = {}

        config["strategy"] = strategy

        # ----------------------------------------------------
        # EMA
        # ----------------------------------------------------

        if "fast_ema" in parameters:

            config["fast_ema"] = (
                parameters[
                    "fast_ema"
                ]
            )

        if "slow_ema" in parameters:

            config["slow_ema"] = (
                parameters[
                    "slow_ema"
                ]
            )

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        if "rsi_period" in parameters:

            config["rsi_period"] = (
                parameters[
                    "rsi_period"
                ]
            )

        if "oversold" in parameters:

            config["oversold"] = (
                parameters[
                    "oversold"
                ]
            )

        if "overbought" in parameters:

            config["overbought"] = (
                parameters[
                    "overbought"
                ]
            )

        # ----------------------------------------------------
        # Supertrend
        # ----------------------------------------------------

        if "atr_period" in parameters:

            config["atr_period"] = (
                parameters[
                    "atr_period"
                ]
            )

        if "multiplier" in parameters:

            config["multiplier"] = (
                parameters[
                    "multiplier"
                ]
            )

        # ----------------------------------------------------
        # Data configuration
        # ----------------------------------------------------

        data_config = strategy_config.get(
            "data",
            {}
        )

        if not isinstance(
            data_config,
            dict
        ):

            data_config = {}

        instrument = (
            data_config.get(
                "instrument"
            )
            or config.get(
                "instrument",
                "EURUSD"
            )
        )

        timeframe = (
            data_config.get(
                "timeframe"
            )
            or config.get(
                "timeframe",
                "1h"
            )
        )

        config["instrument"] = (
            str(
                instrument
            ).upper()
        )

        config["timeframe"] = (
            str(
                timeframe
            ).lower()
        )

        # ----------------------------------------------------
        # Research configuration
        # ----------------------------------------------------

        research_config = (
            strategy_config.get(
                "research",
                {}
            )
        )

        if not isinstance(
            research_config,
            dict
        ):

            research_config = {}

        research_run_id = (
            research_config.get(
                "research_run_id"
            )
            or config.get(
                "research_run_id"
            )
        )

        if not research_run_id:

            raise ValueError(
                "research_run_id is required."
            )

        config[
            "research_run_id"
        ] = str(
            research_run_id
        )

        # ----------------------------------------------------
        # Capital
        # ----------------------------------------------------

        capital_config = (
            strategy_config.get(
                "capital",
                {}
            )
        )

        if not isinstance(
            capital_config,
            dict
        ):

            capital_config = {}

        starting_capital = safe_float(
            capital_config.get(
                "starting_capital",
                config.get(
                    "starting_capital",
                    10000
                )
            ),
            10000
        )

        config[
            "starting_capital"
        ] = starting_capital

        # ----------------------------------------------------
        # Load market data
        # ----------------------------------------------------

        data = load_market_data(
            instrument=
                config["instrument"],

            timeframe=
                config["timeframe"],

            research_run_id=
                config[
                    "research_run_id"
                ]
        )

        # ----------------------------------------------------
        # Deterministic backtest
        # ----------------------------------------------------

        result = run_strategy(
            config,
            data
        )

        # ----------------------------------------------------
        # Chart data
        # ----------------------------------------------------

        chart_data = clean_chart_data(
            result,
            starting_capital
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history = config.get(
            "history",
            []
        )

        if not isinstance(
            history,
            list
        ):

            history = []

        # ----------------------------------------------------
        # Basic response
        # ----------------------------------------------------

        clean_result = {

            "strategy":
                str(
                    result[
                        "strategy"
                    ]
                ),

            "instrument":
                config[
                    "instrument"
                ],

            "timeframe":
                config[
                    "timeframe"
                ],

            "research_run_id":
                config[
                    "research_run_id"
                ],

            "experiment":
                int(
                    config.get(
                        "experiment",
                        1
                    )
                ),

            "max_experiments":
                int(
                    config.get(
                        "max_experiments",
                        5
                    )
                ),

            "history":
                history,

            "strategy_config":
                result[
                    "strategy_config"
                ],

            # ------------------------------------------------
            # Performance
            # ------------------------------------------------

            "initial_capital":
                float(
                    round(
                        result[
                            "initial_capital"
                        ],
                        2
                    )
                ),

            "final_capital":
                float(
                    round(
                        result[
                            "final_capital"
                        ],
                        2
                    )
                ),

            "return_percent":
                float(
                    round(
                        result[
                            "return_percent"
                        ],
                        2
                    )
                ),

            "trades":
                int(
                    result[
                        "trades"
                    ]
                ),

            "win_rate":
                float(
                    round(
                        result[
                            "win_rate"
                        ],
                        2
                    )
                ),

            "profit_factor":
                float(
                    round(
                        result[
                            "profit_factor"
                        ],
                        2
                    )
                ),

            "max_drawdown_percent":
                float(
                    round(
                        result[
                            "max_drawdown_percent"
                        ],
                        2
                    )
                ),

            "sharpe_ratio":
                float(
                    round(
                        result[
                            "sharpe_ratio"
                        ],
                        2
                    )
                ),

            # ------------------------------------------------
            # Heat
            # ------------------------------------------------

            "heat_score":
                int(
                    result[
                        "heat_score"
                    ]
                ),

            "heat_label":
                str(
                    result[
                        "heat_label"
                    ]
                ),

            "heat_checks":
                result[
                    "heat_checks"
                ],

            "baseline_thresholds":
                result[
                    "baseline_thresholds"
                ],

            # ------------------------------------------------
            # REPORT DATA
            # ------------------------------------------------

            "trade_balance_curve":
                chart_data[
                    "trade_balance_curve"
                ],

            "trade_drawdown_curve":
                chart_data[
                    "trade_drawdown_curve"
                ],

            "rolling_sharpe":
                chart_data[
                    "rolling_sharpe"
                ],

            "weekday_analysis":
                chart_data[
                    "weekday_analysis"
                ],

            "stop_loss_exits":
                chart_data[
                    "stop_loss_exits"
                ],

            "take_profit_exits":
                chart_data[
                    "take_profit_exits"
                ],

            "signal_exits":
                chart_data[
                    "signal_exits"
                ],

            "end_of_data_exits":
                chart_data[
                    "end_of_data_exits"
                ]
        }

        # ----------------------------------------------------
        # Strategy-specific fields
        # ----------------------------------------------------

        if strategy == "EMA_Crossover":

            clean_result[
                "fast_ema"
            ] = int(
                result.get(
                    "fast_ema",
                    parameters.get(
                        "fast_ema"
                    )
                )
            )

            clean_result[
                "slow_ema"
            ] = int(
                result.get(
                    "slow_ema",
                    parameters.get(
                        "slow_ema"
                    )
                )
            )

        elif strategy == "RSI_Mean_Reversion":

            clean_result[
                "rsi_period"
            ] = int(
                result.get(
                    "rsi_period",
                    parameters.get(
                        "rsi_period"
                    )
                )
            )

            clean_result[
                "oversold"
            ] = float(
                result.get(
                    "oversold",
                    parameters.get(
                        "oversold"
                    )
                )
            )

            clean_result[
                "overbought"
            ] = float(
                result.get(
                    "overbought",
                    parameters.get(
                        "overbought"
                    )
                )
            )

        elif strategy == "Supertrend":

            clean_result[
                "atr_period"
            ] = int(
                result.get(
                    "atr_period",
                    parameters.get(
                        "atr_period"
                    )
                )
            )

            clean_result[
                "multiplier"
            ] = float(
                result.get(
                    "multiplier",
                    parameters.get(
                        "multiplier"
                    )
                )
            )

        return clean_result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )