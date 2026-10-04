import os
import json

import requests
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPORT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "reports"
)

CHART_DIR = os.path.join(
    REPORT_DIR,
    "charts"
)

API_URL = "http://127.0.0.1:8000/backtest"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def safe_number(value, default=0):
    try:
        if value is None:
            return default

        number = float(value)

        if number != number:
            return default

        return number

    except (TypeError, ValueError):
        return default


def format_number(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}"

    except (TypeError, ValueError):
        return "N/A"


def format_percent(value):
    try:
        return f"{float(value):.2f}%"

    except (TypeError, ValueError):
        return "N/A"


def format_parameter(value):

    if value is None:
        return "None"

    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


def format_bool(value):
    return "PASS" if bool(value) else "FAIL"


def sanitize_filename(value):

    value = str(value).strip()

    invalid_chars = '<>:"/\\|?*'

    for char in invalid_chars:
        value = value.replace(char, "_")

    value = value.replace(" ", "_")

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip(" ._")


def get_heat_from_experiment(experiment):

    if not isinstance(experiment, dict):
        return {
            "score": None,
            "label": "N/A",
            "checks": {}
        }

    score = experiment.get("heat_score")

    label = experiment.get(
        "heat_label",
        "N/A"
    )

    checks = experiment.get(
        "heat_checks",
        {}
    )

    if not isinstance(checks, dict):
        checks = {}

    return {
        "score": score,
        "label": label,
        "checks": checks
    }


# ============================================================
# API
# ============================================================

def run_backtest_through_api(strategy_config):

    payload = {
        "strategy_config": strategy_config
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=120
        )

    except requests.RequestException as error:

        raise RuntimeError(
            f"Could not connect to QuantForge API: {error}"
        )

    if response.status_code != 200:

        raise RuntimeError(
            "QuantForge API returned "
            f"HTTP {response.status_code}:\n"
            f"{response.text}"
        )

    try:

        result = response.json()

    except ValueError:

        raise RuntimeError(
            "QuantForge API did not return valid JSON."
        )

    if not isinstance(result, dict):

        raise RuntimeError(
            "QuantForge API returned an invalid response."
        )

    return result


# ============================================================
# TRADE-DATA NORMALIZATION
# ============================================================

def get_trade_details(backtest_result):

    trades = backtest_result.get(
        "trades_detail",
        []
    )

    if not isinstance(trades, list):
        return []

    return [
        trade
        for trade in trades
        if isinstance(trade, dict)
    ]


def derive_weekday_analysis(trades_detail):

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    results = []

    for day in days:

        day_trades = []

        for trade in trades_detail:

            weekday = trade.get("weekday")

            if weekday == day:
                day_trades.append(trade)

        profits = [
            safe_number(
                trade.get("profit")
            )
            for trade in day_trades
        ]

        winning = [
            profit
            for profit in profits
            if profit > 0
        ]

        if profits:

            results.append({
                "weekday": day,
                "trades": len(profits),
                "profit": sum(profits),
                "average_profit": (
                    sum(profits) / len(profits)
                ),
                "win_rate": (
                    len(winning)
                    / len(profits)
                    * 100
                )
            })

        else:

            results.append({
                "weekday": day,
                "trades": 0,
                "profit": 0.0,
                "average_profit": 0.0,
                "win_rate": 0.0
            })

    return results


def normalize_weekday_analysis(
    backtest_result,
    trades_detail
):

    supplied = backtest_result.get(
        "weekday_analysis"
    )

    if (
        isinstance(supplied, list)
        and len(supplied) > 0
    ):

        return supplied

    return derive_weekday_analysis(
        trades_detail
    )


# ============================================================
# BALANCE CURVE
# ============================================================

def derive_balance_curve(
    trades_detail,
    starting_capital
):

    curve = []

    for index, trade in enumerate(
        trades_detail,
        start=1
    ):

        balance = trade.get(
            "balance_after_trade"
        )

        if balance is None:
            balance = trade.get(
                "balance"
            )

        if balance is None:
            continue

        curve.append({
            "trade_number": index,
            "timestamp": trade.get(
                "exit_time"
            ),
            "balance": safe_number(
                balance
            )
        })

    if curve:
        return curve

    return [{
        "trade_number": 0,
        "timestamp": None,
        "balance": float(
            starting_capital
        )
    }]


def normalize_balance_curve(
    backtest_result,
    trades_detail,
    starting_capital
):

    supplied = backtest_result.get(
        "trade_balance_curve"
    )

    if (
        isinstance(supplied, list)
        and len(supplied) > 1
    ):

        return supplied

    return derive_balance_curve(
        trades_detail,
        starting_capital
    )


# ============================================================
# DRAWDOWN CURVE
# ============================================================

def derive_drawdown_curve(
    trades_detail,
    starting_capital
):

    curve = []

    running_peak = float(
        starting_capital
    )

    for index, trade in enumerate(
        trades_detail,
        start=1
    ):

        balance = trade.get(
            "balance_after_trade"
        )

        if balance is None:
            balance = trade.get(
                "balance"
            )

        if balance is None:
            continue

        balance = safe_number(
            balance,
            running_peak
        )

        running_peak = max(
            running_peak,
            balance
        )

        if running_peak != 0:

            drawdown = (
                (balance - running_peak)
                / running_peak
                * 100
            )

        else:

            drawdown = 0.0

        supplied_drawdown = trade.get(
            "drawdown_percent"
        )

        if supplied_drawdown is not None:

            drawdown = safe_number(
                supplied_drawdown,
                drawdown
            )

        curve.append({
            "trade_number": index,
            "timestamp": trade.get(
                "exit_time"
            ),
            "drawdown_percent": drawdown
        })

    if curve:
        return curve

    return [{
        "trade_number": 0,
        "timestamp": None,
        "drawdown_percent": 0.0
    }]


def normalize_drawdown_curve(
    backtest_result,
    trades_detail,
    starting_capital
):

    supplied = backtest_result.get(
        "trade_drawdown_curve"
    )

    if (
        isinstance(supplied, list)
        and len(supplied) > 1
    ):

        return supplied

    return derive_drawdown_curve(
        trades_detail,
        starting_capital
    )


# ============================================================
# ROLLING SHARPE
# ============================================================

def derive_trade_returns(
    trades_detail,
    starting_capital
):

    returns = []

    balance = float(
        starting_capital
    )

    for trade in trades_detail:

        profit = safe_number(
            trade.get("profit")
        )

        if balance != 0:

            trade_return = (
                profit / balance
            )

        else:

            trade_return = 0.0

        returns.append(
            float(trade_return)
        )

        balance += profit

    return returns


def calculate_rolling_sharpe_values(
    returns,
    window=20
):

    values = []

    for i in range(
        len(returns)
    ):

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

        mean_value = (
            sum(window_returns)
            / len(window_returns)
        )

        variance = sum(
            (
                value - mean_value
            ) ** 2
            for value in window_returns
        ) / (
            len(window_returns) - 1
        )

        std = variance ** 0.5

        if std == 0:

            values.append(0.0)
            continue

        sharpe = (
            mean_value
            / std
            * (252 ** 0.5)
        )

        values.append(
            float(sharpe)
        )

    return values


def derive_rolling_sharpe(
    trades_detail,
    starting_capital
):

    returns = derive_trade_returns(
        trades_detail,
        starting_capital
    )

    values = calculate_rolling_sharpe_values(
        returns,
        window=20
    )

    result = []

    for index, value in enumerate(
        values
    ):

        timestamp = None

        if index < len(trades_detail):

            timestamp = trades_detail[
                index
            ].get("exit_time")

        result.append({
            "trade_number": index + 1,
            "timestamp": timestamp,
            "sharpe_ratio": float(value)
        })

    return result


def normalize_rolling_sharpe(
    backtest_result,
    trades_detail,
    starting_capital
):

    supplied = backtest_result.get(
        "rolling_sharpe"
    )

    if (
        isinstance(supplied, list)
        and len(supplied) > 1
    ):

        return supplied

    return derive_rolling_sharpe(
        trades_detail,
        starting_capital
    )


# ============================================================
# EXIT STATISTICS
# ============================================================

def derive_exit_counts(
    trades_detail
):

    counts = {
        "stop_loss_exits": 0,
        "take_profit_exits": 0,
        "signal_exits": 0,
        "end_of_data_exits": 0
    }

    for trade in trades_detail:

        reason = str(
            trade.get(
                "exit_reason",
                ""
            )
        ).lower()

        if reason == "stop_loss":

            counts[
                "stop_loss_exits"
            ] += 1

        elif reason == "take_profit":

            counts[
                "take_profit_exits"
            ] += 1

        elif reason == "signal":

            counts[
                "signal_exits"
            ] += 1

        elif reason == "end_of_data":

            counts[
                "end_of_data_exits"
            ] += 1

    return counts


def get_exit_counts(
    backtest_result,
    trades_detail
):

    derived = derive_exit_counts(
        trades_detail
    )

    result = {}

    for key in derived:

        supplied = backtest_result.get(
            key
        )

        if supplied is None:

            result[key] = derived[key]

        else:

            supplied_number = int(
                safe_number(supplied)
            )

            if (
                supplied_number == 0
                and derived[key] > 0
            ):

                result[key] = derived[key]

            else:

                result[key] = supplied_number

    return result


# ============================================================
# CHART — WEEKDAY
# ============================================================

def create_weekday_chart(
    weekday_analysis
):

    path = os.path.join(
        CHART_DIR,
        "weekday_analysis.png"
    )

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    profit_map = {}

    for item in weekday_analysis:

        if not isinstance(
            item,
            dict
        ):
            continue

        day = item.get(
            "weekday"
        )

        if day:

            profit_map[
                day
            ] = safe_number(
                item.get("profit")
            )

    profits = [
        profit_map.get(
            day,
            0
        )
        for day in days
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        days,
        profits
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Profit by Weekday"
    )

    plt.xlabel(
        "Weekday"
    )

    plt.ylabel(
        "Profit"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return path


# ============================================================
# CHART — SHARPE
# ============================================================

def create_sharpe_chart(
    rolling_sharpe
):

    path = os.path.join(
        CHART_DIR,
        "sharpe_ratio.png"
    )

    values = []

    for item in rolling_sharpe:

        if isinstance(
            item,
            dict
        ):

            value = item.get(
                "sharpe_ratio"
            )

            if value is None:

                value = item.get(
                    "sharpe"
                )

            if value is None:

                value = item.get(
                    "rolling_sharpe"
                )

            values.append(
                safe_number(value)
            )

        else:

            values.append(
                safe_number(item)
            )

    plt.figure(
        figsize=(10, 5)
    )

    if values:

        plt.plot(
            range(
                1,
                len(values) + 1
            ),
            values
        )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Rolling Sharpe Ratio"
    )

    plt.xlabel(
        "Trade Number"
    )

    plt.ylabel(
        "Sharpe Ratio"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return path


# ============================================================
# CHART — DRAWDOWN
# ============================================================

def create_drawdown_chart(
    trade_drawdown_curve
):

    path = os.path.join(
        CHART_DIR,
        "drawdown.png"
    )

    values = []

    for item in trade_drawdown_curve:

        if isinstance(
            item,
            dict
        ):

            value = item.get(
                "drawdown_percent"
            )

            if value is None:

                value = item.get(
                    "drawdown"
                )

            values.append(
                safe_number(value)
            )

        else:

            values.append(
                safe_number(item)
            )

    plt.figure(
        figsize=(10, 5)
    )

    if values:

        plt.plot(
            range(
                1,
                len(values) + 1
            ),
            values
        )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.title(
        "Drawdown Across Trades"
    )

    plt.xlabel(
        "Trade Number"
    )

    plt.ylabel(
        "Drawdown (%)"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return path


# ============================================================
# CHART — BALANCE
# ============================================================

def create_balance_chart(
    trade_balance_curve
):

    path = os.path.join(
        CHART_DIR,
        "balance_per_trade.png"
    )

    values = []

    for item in trade_balance_curve:

        if isinstance(
            item,
            dict
        ):

            value = item.get(
                "balance_after_trade"
            )

            if value is None:

                value = item.get(
                    "balance"
                )

            values.append(
                safe_number(value)
            )

        else:

            values.append(
                safe_number(item)
            )

    plt.figure(
        figsize=(10, 5)
    )

    if values:

        plt.plot(
            range(
                1,
                len(values) + 1
            ),
            values
        )

    plt.title(
        "Balance After Every Trade"
    )

    plt.xlabel(
        "Trade Number"
    )

    plt.ylabel(
        "Account Balance"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    return path


# ============================================================
# TABLE STYLE
# ============================================================

def standard_table_style(
    header=True,
    font_size=8,
    padding=6
):

    commands = [
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            font_size
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "TOP"
        ),

        (
            "PADDING",
            (0, 0),
            (-1, -1),
            padding
        )
    ]

    if header:

        commands.extend([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            )
        ])

    return TableStyle(
        commands
    )


# ============================================================
# MAIN REPORT
# ============================================================

def generate_report(payload):

    if not isinstance(
        payload,
        dict
    ):

        raise ValueError(
            "Report payload must be a JSON object."
        )

    strategy_config = payload.get(
        "strategy_config"
    )

    if not isinstance(
        strategy_config,
        dict
    ):

        raise ValueError(
            "strategy_config is missing."
        )

    strategy = (
        payload.get("strategy")
        or strategy_config.get(
            "strategy"
        )
    )

    if not strategy:

        raise ValueError(
            "Selected strategy is missing."
        )

    parameters = strategy_config.get(
        "parameters",
        {}
    )

    risk = strategy_config.get(
        "risk",
        {}
    )

    data_config = strategy_config.get(
        "data",
        {}
    )

    capital_config = strategy_config.get(
        "capital",
        {}
    )

    if not isinstance(
        parameters,
        dict
    ):
        parameters = {}

    if not isinstance(
        risk,
        dict
    ):
        risk = {}

    if not isinstance(
        data_config,
        dict
    ):
        data_config = {}

    if not isinstance(
        capital_config,
        dict
    ):
        capital_config = {}

    instrument = (
        payload.get("instrument")
        or data_config.get(
            "instrument"
        )
        or "EURUSD"
    )

    timeframe = (
        payload.get("timeframe")
        or data_config.get(
            "timeframe"
        )
        or "1h"
    )

    starting_capital = safe_number(
        payload.get(
            "starting_capital"
        )
        or capital_config.get(
            "starting_capital"
        )
        or 10000
    )

    research_run_id = (
        payload.get(
            "research_run_id"
        )
        or strategy_config.get(
            "research",
            {}
        ).get(
            "research_run_id",
            "N/A"
        )
    )

    history = payload.get(
        "history",
        []
    )

    if not isinstance(
        history,
        list
    ):
        history = []

    best_experiment = payload.get(
        "best_experiment",
        "N/A"
    )

    best_experiment_data = None

    for experiment in history:

        if not isinstance(
            experiment,
            dict
        ):
            continue

        if str(
            experiment.get(
                "experiment"
            )
        ) == str(
            best_experiment
        ):

            best_experiment_data = (
                experiment
            )

            break

    if (
        best_experiment_data is None
        and history
    ):

        best_experiment_data = (
            history[-1]
        )

    if not isinstance(
        best_experiment_data,
        dict
    ):

        best_experiment_data = {}

    final_heat = get_heat_from_experiment(
        best_experiment_data
    )

    final_heat_score = final_heat[
        "score"
    ]

    final_heat_label = final_heat[
        "label"
    ]

    final_heat_checks = final_heat[
        "checks"
    ]

    validation = payload.get(
        "validation",
        "UNKNOWN"
    )

    validation_checks = payload.get(
        "validation_checks",
        {}
    )

    if not isinstance(
        validation_checks,
        dict
    ):
        validation_checks = {}

    validation_reason = payload.get(
        "validation_reason",
        "No validator explanation provided."
    )

    minimum_trades_required = payload.get(
        "minimum_trades_required",
        100
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    backtest_result = (
        run_backtest_through_api(
            strategy_config
        )
    )

    # ========================================================
    # BASIC METRICS
    # ========================================================

    return_percent = safe_number(
        backtest_result.get(
            "return_percent"
        )
    )

    trades = int(
        safe_number(
            backtest_result.get(
                "trades"
            )
        )
    )

    win_rate = safe_number(
        backtest_result.get(
            "win_rate"
        )
    )

    profit_factor = safe_number(
        backtest_result.get(
            "profit_factor"
        )
    )

    max_drawdown = safe_number(
        backtest_result.get(
            "max_drawdown_percent"
        )
    )

    sharpe_ratio = safe_number(
        backtest_result.get(
            "sharpe_ratio"
        )
    )

    # ========================================================
    # NORMALIZE TRADE DATA
    # ========================================================

    trades_detail = get_trade_details(
        backtest_result
    )

    weekday_analysis = (
        normalize_weekday_analysis(
            backtest_result,
            trades_detail
        )
    )

    rolling_sharpe = (
        normalize_rolling_sharpe(
            backtest_result,
            trades_detail,
            starting_capital
        )
    )

    trade_drawdown_curve = (
        normalize_drawdown_curve(
            backtest_result,
            trades_detail,
            starting_capital
        )
    )

    trade_balance_curve = (
        normalize_balance_curve(
            backtest_result,
            trades_detail,
            starting_capital
        )
    )

    exit_counts = get_exit_counts(
        backtest_result,
        trades_detail
    )

    # ========================================================
    # CREATE CHARTS
    # ========================================================

    weekday_chart = (
        create_weekday_chart(
            weekday_analysis
        )
    )

    sharpe_chart = (
        create_sharpe_chart(
            rolling_sharpe
        )
    )

    drawdown_chart = (
        create_drawdown_chart(
            trade_drawdown_curve
        )
    )

    balance_chart = (
        create_balance_chart(
            trade_balance_curve
        )
    )

    # ========================================================
    # FILENAME
    # ========================================================

    safe_instrument = sanitize_filename(
        instrument
    )

    safe_timeframe = sanitize_filename(
        timeframe
    )

    safe_strategy = sanitize_filename(
        strategy
    )

    report_filename = (
        f"{safe_instrument}_"
        f"{safe_timeframe}_"
        f"{safe_strategy}.pdf"
    )

    report_path = os.path.join(
        REPORT_DIR,
        report_filename
    )

    # ========================================================
    # STYLES
    # ========================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    subheading_style = ParagraphStyle(
        "ReportSubHeading",
        parent=styles["Heading3"],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=6
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=7,
        leading=10
    )

    document = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    story = []

    # ========================================================
    # PAGE 1
    # ========================================================

    story.append(
        Paragraph(
            "QuantForge Strategy Research Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Autonomous Forex Strategy Research and Validation",
            body_style
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "1. Research Objective",
            heading_style
        )
    )

    # Intentionally preserve current behavior:
    # objective belongs to the front-end/user-input pipeline.
    story.append(
        Paragraph(
            str(
                payload.get(
                    "research_objective",
                    "Not provided"
                )
            ),
            body_style
        )
    )

    story.append(
        Paragraph(
            "2. Chosen Strategy",
            heading_style
        )
    )

    strategy_rows = [
        ["Field", "Value"],
        ["Strategy", strategy],
        ["Instrument", instrument],
        ["Timeframe", timeframe],
        [
            "Starting Capital",
            format_number(
                starting_capital
            )
        ],
        [
            "Selected Experiment",
            str(best_experiment)
        ],
        [
            "Research Run ID",
            str(research_run_id)
        ]
    ]

    strategy_table = Table(
        strategy_rows,
        colWidths=[
            2.2 * inch,
            3.8 * inch
        ]
    )

    strategy_table.setStyle(
        standard_table_style()
    )

    story.append(
        strategy_table
    )

    story.append(
        Paragraph(
            "Strategy Parameters",
            heading_style
        )
    )

    parameter_rows = [
        ["Parameter", "Value"]
    ]

    for key, value in parameters.items():

        parameter_rows.append([
            str(key),
            format_parameter(value)
        ])

    parameter_table = Table(
        parameter_rows,
        colWidths=[
            2.2 * inch,
            3.8 * inch
        ]
    )

    parameter_table.setStyle(
        standard_table_style()
    )

    story.append(
        parameter_table
    )

    story.append(
        Paragraph(
            "Risk Configuration",
            heading_style
        )
    )

    risk_rows = [
        ["Risk Parameter", "Value"],
        [
            "Stop Loss",
            format_parameter(
                risk.get("stop_loss")
            )
        ],
        [
            "Take Profit",
            format_parameter(
                risk.get("take_profit")
            )
        ],
        [
            "Lot Size",
            format_parameter(
                risk.get("lot_size")
            )
        ]
    ]

    risk_table = Table(
        risk_rows,
        colWidths=[
            2.2 * inch,
            3.8 * inch
        ]
    )

    risk_table.setStyle(
        standard_table_style()
    )

    story.append(
        risk_table
    )

    story.append(
        Paragraph(
            "3. Final Performance",
            heading_style
        )
    )

    performance_rows = [
        ["Metric", "Result"],
        [
            "Return",
            format_percent(
                return_percent
            )
        ],
        [
            "Trades",
            str(trades)
        ],
        [
            "Win Rate",
            format_percent(
                win_rate
            )
        ],
        [
            "Profit Factor",
            format_number(
                profit_factor
            )
        ],
        [
            "Maximum Drawdown",
            format_percent(
                max_drawdown
            )
        ],
        [
            "Sharpe Ratio",
            format_number(
                sharpe_ratio
            )
        ],
        [
            "Heat Score",
            (
                f"{final_heat_score}/4"
                if final_heat_score is not None
                else "N/A"
            )
        ],
        [
            "Heat Label",
            str(final_heat_label)
        ],
        [
            "Validation",
            str(validation)
        ]
    ]

    performance_table = Table(
        performance_rows,
        colWidths=[
            2.8 * inch,
            3.2 * inch
        ]
    )

    performance_table.setStyle(
        standard_table_style()
    )

    story.append(
        performance_table
    )

    story.append(
        Paragraph(
            "4. Deterministic Heat Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The Heat Score is generated by the deterministic "
            "Python backtesting engine. It is based on four "
            "checks: positive return, profit factor above 1, "
            "positive Sharpe ratio, and maximum drawdown "
            "better than -10%.",
            body_style
        )
    )

    final_heat_rows = [
        ["Heat Field", "Result"],
        [
            "Heat Score",
            (
                f"{final_heat_score}/4"
                if final_heat_score is not None
                else "N/A"
            )
        ],
        [
            "Heat Label",
            str(final_heat_label)
        ],
        [
            "Positive Return",
            format_bool(
                final_heat_checks.get(
                    "positive_return",
                    False
                )
            )
        ],
        [
            "Profit Factor > 1",
            format_bool(
                final_heat_checks.get(
                    "profit_factor_above_1",
                    False
                )
            )
        ],
        [
            "Sharpe Ratio > 0",
            format_bool(
                final_heat_checks.get(
                    "positive_sharpe",
                    False
                )
            )
        ],
        [
            "Drawdown > -10%",
            format_bool(
                final_heat_checks.get(
                    "drawdown_better_than_10_percent",
                    False
                )
            )
        ]
    ]

    final_heat_table = Table(
        final_heat_rows,
        colWidths=[
            3.0 * inch,
            3.0 * inch
        ]
    )

    final_heat_table.setStyle(
        standard_table_style()
    )

    story.append(
        final_heat_table
    )

    # ========================================================
    # HEAT PROGRESSION
    # ========================================================

    story.append(
        Paragraph(
            "Heat Progression Across Experiments",
            subheading_style
        )
    )

    heat_rows = [[
        "Experiment",
        "Strategy",
        "Heat Score",
        "Heat Label",
        "Return",
        "PF",
        "Sharpe",
        "DD"
    ]]

    for index, experiment in enumerate(
        history,
        start=1
    ):

        if not isinstance(
            experiment,
            dict
        ):
            continue

        heat = get_heat_from_experiment(
            experiment
        )

        config = experiment.get(
            "strategy_config",
            {}
        )

        if not isinstance(
            config,
            dict
        ):
            config = {}

        exp_strategy = (
            experiment.get(
                "strategy"
            )
            or config.get(
                "strategy",
                "N/A"
            )
        )

        heat_rows.append([
            str(
                experiment.get(
                    "experiment",
                    index
                )
            ),
            str(exp_strategy),
            (
                f"{heat['score']}/4"
                if heat["score"] is not None
                else "N/A"
            ),
            str(
                heat["label"]
            ),
            format_percent(
                experiment.get(
                    "return_percent"
                )
            ),
            format_number(
                experiment.get(
                    "profit_factor"
                )
            ),
            format_number(
                experiment.get(
                    "sharpe_ratio"
                )
            ),
            format_percent(
                experiment.get(
                    "max_drawdown_percent"
                )
            )
        ])

    heat_table = Table(
        heat_rows,
        colWidths=[
            0.55 * inch,
            1.1 * inch,
            0.75 * inch,
            1.0 * inch,
            0.7 * inch,
            0.55 * inch,
            0.65 * inch,
            0.7 * inch
        ],
        repeatRows=1
    )

    heat_table.setStyle(
        standard_table_style(
            font_size=7,
            padding=4
        )
    )

    story.append(
        heat_table
    )

    if len(history) > 1:

        story.append(
            Paragraph(
                "Heat Changes",
                subheading_style
            )
        )

        heat_change_rows = [[
            "Transition",
            "Previous",
            "Current",
            "Change"
        ]]

        for index in range(
            1,
            len(history)
        ):

            previous = history[
                index - 1
            ]

            current = history[
                index
            ]

            previous_heat = (
                get_heat_from_experiment(
                    previous
                )
            )

            current_heat = (
                get_heat_from_experiment(
                    current
                )
            )

            previous_score = (
                previous_heat["score"]
            )

            current_score = (
                current_heat["score"]
            )

            if (
                previous_score is not None
                and current_score is not None
            ):

                change = (
                    float(current_score)
                    - float(previous_score)
                )

                if change > 0:

                    change_text = (
                        f"+{change:g}"
                    )

                else:

                    change_text = (
                        f"{change:g}"
                    )

            else:

                change_text = "N/A"

            heat_change_rows.append([
                (
                    f"Experiment "
                    f"{previous.get('experiment', index)}"
                    f" → Experiment "
                    f"{current.get('experiment', index + 1)}"
                ),
                (
                    f"{previous_score}/4"
                    if previous_score is not None
                    else "N/A"
                ),
                (
                    f"{current_score}/4"
                    if current_score is not None
                    else "N/A"
                ),
                change_text
            ])

        heat_change_table = Table(
            heat_change_rows,
            colWidths=[
                2.5 * inch,
                1.0 * inch,
                1.0 * inch,
                1.0 * inch
            ],
            repeatRows=1
        )

        heat_change_table.setStyle(
            standard_table_style(
                font_size=8,
                padding=5
            )
        )

        story.append(
            heat_change_table
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    story.append(
        Paragraph(
            "5. Deterministic Validation",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The validation result is taken directly "
            "from the deterministic validation stage.",
            body_style
        )
    )

    validation_rows = [
        ["Validation Check", "Result"]
    ]

    for key, value in validation_checks.items():

        label = (
            str(key)
            .replace("_", " ")
            .title()
        )

        validation_rows.append([
            label,
            "PASS" if value else "FAIL"
        ])

    validation_table = Table(
        validation_rows,
        colWidths=[
            4.0 * inch,
            2.0 * inch
        ]
    )

    validation_table.setStyle(
        standard_table_style()
    )

    story.append(
        validation_table
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            f"Minimum trades required: "
            f"{minimum_trades_required}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Validator Explanation:</b> "
            f"{str(validation_reason)}",
            body_style
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # EXPERIMENT SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "6. Experiment Summary",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"Experiments conducted: "
            f"{len(history)}",
            body_style
        )
    )

    experiment_rows = [[
        "Experiment",
        "Strategy",
        "Parameters",
        "Return",
        "Trades",
        "PF",
        "Sharpe",
        "DD",
        "Heat"
    ]]

    for index, experiment in enumerate(
        history,
        start=1
    ):

        if not isinstance(
            experiment,
            dict
        ):
            continue

        config = experiment.get(
            "strategy_config",
            {}
        )

        if not isinstance(
            config,
            dict
        ):
            config = {}

        exp_strategy = (
            config.get(
                "strategy"
            )
            or experiment.get(
                "strategy",
                "Unknown"
            )
        )

        exp_parameters = config.get(
            "parameters",
            {}
        )

        if isinstance(
            exp_parameters,
            dict
        ):

            parameter_text = ", ".join(
                f"{key}={value}"
                for key, value
                in exp_parameters.items()
            )

        else:

            parameter_text = str(
                exp_parameters
            )

        heat = get_heat_from_experiment(
            experiment
        )

        heat_text = (
            f"{heat['score']}/4\n"
            f"{heat['label']}"
            if heat["score"] is not None
            else "N/A"
        )

        experiment_rows.append([
            str(
                experiment.get(
                    "experiment",
                    index
                )
            ),
            str(exp_strategy),
            Paragraph(
                parameter_text,
                small_style
            ),
            format_percent(
                experiment.get(
                    "return_percent"
                )
            ),
            str(
                experiment.get(
                    "trades",
                    "N/A"
                )
            ),
            format_number(
                experiment.get(
                    "profit_factor"
                )
            ),
            format_number(
                experiment.get(
                    "sharpe_ratio"
                )
            ),
            format_percent(
                experiment.get(
                    "max_drawdown_percent"
                )
            ),
            Paragraph(
                heat_text,
                small_style
            )
        ])

    experiment_table = Table(
        experiment_rows,
        colWidths=[
            0.5 * inch,
            0.85 * inch,
            1.65 * inch,
            0.6 * inch,
            0.5 * inch,
            0.5 * inch,
            0.55 * inch,
            0.65 * inch,
            0.85 * inch
        ],
        repeatRows=1
    )

    experiment_table.setStyle(
        standard_table_style(
            font_size=7,
            padding=4
        )
    )

    story.append(
        experiment_table
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # WEEKDAY ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "7. Weekday Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The weekday analysis summarizes completed "
            "trades according to the weekday on which "
            "the trade exited.",
            body_style
        )
    )

    story.append(
        Image(
            weekday_chart,
            width=6.7 * inch,
            height=3.35 * inch
        )
    )

    weekday_rows = [[
        "Weekday",
        "Trades",
        "Profit",
        "Average Profit",
        "Win Rate"
    ]]

    for item in weekday_analysis:

        if not isinstance(
            item,
            dict
        ):
            continue

        weekday_rows.append([
            str(
                item.get(
                    "weekday",
                    "N/A"
                )
            ),
            str(
                item.get(
                    "trades",
                    0
                )
            ),
            format_number(
                item.get(
                    "profit"
                )
            ),
            format_number(
                item.get(
                    "average_profit"
                )
            ),
            format_percent(
                item.get(
                    "win_rate"
                )
            )
        ])

    weekday_table = Table(
        weekday_rows,
        colWidths=[
            1.3 * inch,
            0.8 * inch,
            1.4 * inch,
            1.4 * inch,
            1.1 * inch
        ]
    )

    weekday_table.setStyle(
        standard_table_style()
    )

    story.append(
        weekday_table
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # SHARPE
    # ========================================================

    story.append(
        Paragraph(
            "8. Sharpe Ratio Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The chart shows the rolling Sharpe ratio "
            "across completed trades.",
            body_style
        )
    )

    story.append(
        Image(
            sharpe_chart,
            width=6.7 * inch,
            height=3.35 * inch
        )
    )

    story.append(
        Paragraph(
            f"Final Sharpe ratio: "
            f"{format_number(sharpe_ratio)}",
            body_style
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    story.append(
        Paragraph(
            "9. Drawdown Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The drawdown chart shows account drawdown "
            "after each completed trade.",
            body_style
        )
    )

    story.append(
        Image(
            drawdown_chart,
            width=6.7 * inch,
            height=3.35 * inch
        )
    )

    story.append(
        Paragraph(
            f"Maximum drawdown: "
            f"{format_percent(max_drawdown)}",
            body_style
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # BALANCE + EXIT DISTRIBUTION
    # ========================================================

    story.append(
        Paragraph(
            "10. Balance After Every Trade",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The balance curve tracks the account balance "
            "after every completed trade.",
            body_style
        )
    )

    story.append(
        Image(
            balance_chart,
            width=6.7 * inch,
            height=3.35 * inch
        )
    )

    story.append(
        Paragraph(
            "Exit Distribution",
            heading_style
        )
    )

    exit_rows = [
        [
            "Exit Reason",
            "Number of Trades"
        ],
        [
            "Stop Loss",
            str(
                exit_counts[
                    "stop_loss_exits"
                ]
            )
        ],
        [
            "Take Profit",
            str(
                exit_counts[
                    "take_profit_exits"
                ]
            )
        ],
        [
            "Signal Exit",
            str(
                exit_counts[
                    "signal_exits"
                ]
            )
        ],
        [
            "End of Data",
            str(
                exit_counts[
                    "end_of_data_exits"
                ]
            )
        ]
    ]

    exit_table = Table(
        exit_rows,
        colWidths=[
            3.0 * inch,
            2.0 * inch
        ]
    )

    exit_table.setStyle(
        standard_table_style()
    )

    story.append(
        exit_table
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # CONCLUSION
    # ========================================================

    story.append(
        Paragraph(
            "11. Research Conclusion",
            heading_style
        )
    )

    conclusion = (
        f"Experiment {best_experiment} was selected "
        f"as the best candidate using the "
        f"{strategy} strategy. The selected configuration "
        f"achieved a return of "
        f"{format_percent(return_percent)}, "
        f"a profit factor of "
        f"{format_number(profit_factor)}, "
        f"a Sharpe ratio of "
        f"{format_number(sharpe_ratio)}, "
        f"and maximum drawdown of "
        f"{format_percent(max_drawdown)}. "
    )

    if final_heat_score is not None:

        conclusion += (
            f"The deterministic Heat Score was "
            f"{final_heat_score}/4 "
            f"({final_heat_label}). "
        )

    conclusion += (
        f"The deterministic validation stage marked "
        f"the strategy as {validation}."
    )

    story.append(
        Paragraph(
            conclusion,
            body_style
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "This report represents strategy research output "
            "from the QuantForge experimental process. "
            "It is not a live trading recommendation.",
            body_style
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )

    return {
        "report_path": report_path,
        "report_filename": report_filename,
        "strategy": strategy,
        "instrument": instrument,
        "timeframe": timeframe,
        "best_experiment": best_experiment,
        "heat_score": final_heat_score,
        "heat_label": final_heat_label,
        "experiments": len(history),
        "trades_used_for_charts": trades
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage: python report_generator.py payload.json"
        )

        raise SystemExit(1)

    input_file = sys.argv[1]

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as file:

        payload = json.load(file)

    if isinstance(
        payload,
        list
    ):

        if not payload:

            raise ValueError(
                "Input JSON array is empty."
            )

        payload = payload[0]

    result = generate_report(
        payload
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )