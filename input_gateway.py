import json
import os
import re
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yfinance as yf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


# ============================================================
# QUANTFORGE CONFIGURATION
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

N8N_WEBHOOK_URL = (
    "http://127.0.0.1:5678/webhook/quantforge-research"
)

PROGRESS_SERVER_URL = (
    "http://127.0.0.1:8002"
)

PROJECT_DIRECTORY = Path(
    r"C:\Users\aj\Documents\QuantForge"
)

REPORT_DIRECTORY = (
    PROJECT_DIRECTORY /
    "quant_engine" /
    "reports"
)

MIN_STARTING_CAPITAL = 1000
MAX_STARTING_CAPITAL = 100_000_000

MIN_EXPERIMENTS = 1
MAX_EXPERIMENTS = 20

N8N_TIMEOUT_SECONDS = 3600


# ============================================================
# SUPPORTED FOREX INSTRUMENTS
# ============================================================

SUPPORTED_FOREX = {
    "EURUSD": {
        "name": "EUR/USD",
        "yahoo": "EURUSD=X",
    },
    "GBPUSD": {
        "name": "GBP/USD",
        "yahoo": "GBPUSD=X",
    },
    "USDJPY": {
        "name": "USD/JPY",
        "yahoo": "JPY=X",
    },
    "USDCHF": {
        "name": "USD/CHF",
        "yahoo": "CHF=X",
    },
    "AUDUSD": {
        "name": "AUD/USD",
        "yahoo": "AUDUSD=X",
    },
    "USDCAD": {
        "name": "USD/CAD",
        "yahoo": "CAD=X",
    },
    "NZDUSD": {
        "name": "NZD/USD",
        "yahoo": "NZDUSD=X",
    },
}


# ============================================================
# SUPPORTED TIMEFRAMES
# ============================================================

ALLOWED_TIMEFRAMES = {
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
}


# ============================================================
# STRATEGIES
# ============================================================

SUPPORTED_STRATEGIES = {
    "ema_crossover",
    "ema crossover",
    "ema",
    "rsi_mean_reversion",
    "rsi mean reversion",
    "rsi",
    "supertrend",
}

UNSUPPORTED_STRATEGIES = {
    "macd",
    "moving average convergence divergence",
    "bollinger",
    "bollinger bands",
    "stochastic",
    "stochastic oscillator",
    "adx",
    "average directional index",
    "ichimoku",
    "ichimoku cloud",
    "vwap",
    "cci",
    "commodity channel index",
    "parabolic sar",
    "parabolic",
    "donchian",
    "donchian channels",
}


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="QuantForge Input Gateway",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ResearchRequest(BaseModel):
    request: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    timeframe: str = Field(
        default="15m"
    )

    starting_capital: float = Field(
        default=10000,
        gt=0,
    )

    max_experiments: int = Field(
        default=5,
        ge=1,
        le=20,
    )


# ============================================================
# HELPERS
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:

    text = str(text).strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end > start:

            return json.loads(
                text[start:end + 1]
            )

    raise ValueError(
        "The LLM did not return valid JSON."
    )


def generate_research_run_id() -> str:

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    short_id = uuid.uuid4().hex[:8]

    return (
        f"dashboard_{timestamp}_{short_id}"
    )


def normalize_text(
    value: Optional[str]
) -> str:

    if value is None:
        return ""

    value = str(value).strip().lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


# ============================================================
# OLLAMA INTENT CLASSIFIER
# ============================================================

def ask_ollama(
    user_request: str
) -> Dict[str, Any]:

    prompt = f"""
You are QuantForge's research-intent classifier.

QuantForge is a financial-market strategy research system.

It researches trading strategies using historical market data.

It is NOT a live trading system.

Determine whether the user's actual request is asking
QuantForge to research a financial-market trading strategy.

Judge meaning, not keywords.

VALID examples:

Find a trading strategy for EUR/USD.
Research an intraday strategy for EURUSD.
Find a mean-reversion strategy for GBP/USD.
Test strategies for USD/JPY.
Research a low-drawdown strategy for EUR/USD.

INVALID examples:

Book a ticket to Goa.
Plan my vacation.
Find a strategy for buying a house.
Tell me what restaurant to visit.
Write a Python calculator.
Explain what RSI means.

QuantForge currently supports Forex research.

Supported instruments:

EUR/USD
GBP/USD
USD/JPY
USD/CHF
AUD/USD
USD/CAD
NZD/USD

Supported strategy families:

EMA_Crossover
RSI_Mean_Reversion
Supertrend

If the user explicitly requests another strategy,
mark the request invalid.

Do NOT invent a ticker.

Do NOT resolve Yahoo Finance symbols.

Python performs symbol resolution.

Do NOT invent a timeframe.

The UI supplies the timeframe separately.

Return ONLY JSON.

Required structure:

{{
  "request_valid": true,
  "reason": "short reason",
  "research_objective": "short research objective",
  "asset_name": "asset/instrument mentioned",
  "user_symbol": "explicit symbol if present, otherwise null",
  "market": "Forex if clear, otherwise null",
  "requested_strategy": "strategy if explicitly requested, otherwise null"
}}

User request:

{user_request}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 300,
            },
        },
        timeout=180,
    )

    response.raise_for_status()

    body = response.json()

    return extract_json(
        body["response"]
    )


# ============================================================
# ASSET RESOLUTION
# ============================================================

ASSET_ALIASES = {

    "eur/usd": "EURUSD",
    "eurusd": "EURUSD",
    "euro dollar": "EURUSD",
    "euro usd": "EURUSD",

    "gbp/usd": "GBPUSD",
    "gbpusd": "GBPUSD",
    "pound dollar": "GBPUSD",
    "pound usd": "GBPUSD",

    "usd/jpy": "USDJPY",
    "usdjpy": "USDJPY",
    "dollar yen": "USDJPY",

    "usd/chf": "USDCHF",
    "usdchf": "USDCHF",
    "dollar franc": "USDCHF",

    "aud/usd": "AUDUSD",
    "audusd": "AUDUSD",
    "australian dollar": "AUDUSD",

    "usd/cad": "USDCAD",
    "usdcad": "USDCAD",
    "dollar canadian": "USDCAD",

    "nzd/usd": "NZDUSD",
    "nzdusd": "NZDUSD",
    "new zealand dollar": "NZDUSD",
}


def resolve_instrument(
    parsed: Dict[str, Any]
) -> Optional[str]:

    candidates = [
        parsed.get("user_symbol"),
        parsed.get("asset_name"),
    ]

    for candidate in candidates:

        value = normalize_text(candidate)

        if not value:
            continue

        if value in ASSET_ALIASES:
            return ASSET_ALIASES[value]

        compact = value.replace(
            " ",
            "",
        )

        if compact in ASSET_ALIASES:
            return ASSET_ALIASES[compact]

        upper = value.upper()

        for instrument, metadata in SUPPORTED_FOREX.items():

            if upper == instrument:
                return instrument

            if upper == metadata["yahoo"]:
                return instrument

    return None


# ============================================================
# YAHOO VERIFICATION
# ============================================================

def verify_yahoo_symbol(
    instrument: str
) -> Dict[str, Any]:

    metadata = SUPPORTED_FOREX.get(
        instrument
    )

    if not metadata:

        return {
            "verified": False,
            "message": (
                f"{instrument} is not a supported "
                "QuantForge Forex instrument."
            ),
        }

    symbol = metadata["yahoo"]

    try:

        ticker = yf.Ticker(symbol)

        end = datetime.now()

        start = (
            end -
            timedelta(days=14)
        )

        data = ticker.history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
        )

        if data is None or data.empty:

            return {
                "verified": False,
                "symbol": symbol,
                "message": (
                    f"Yahoo Finance returned no "
                    f"usable data for {symbol}."
                ),
            }

        return {
            "verified": True,
            "instrument": instrument,
            "symbol": symbol,
            "asset_name": metadata["name"],
            "rows": int(len(data)),
            "message": (
                f"Yahoo Finance data is available "
                f"for {metadata['name']} ({symbol})."
            ),
        }

    except Exception as error:

        return {
            "verified": False,
            "symbol": symbol,
            "message": (
                f"Could not verify {symbol} "
                f"with Yahoo Finance: {error}"
            ),
        }


# ============================================================
# STRATEGY VALIDATION
# ============================================================

def detect_explicit_unsupported_strategy(
    user_request: str
) -> Optional[str]:

    text = normalize_text(
        user_request
    )

    for strategy in UNSUPPORTED_STRATEGIES:

        if strategy in text:
            return strategy

    return None


def check_strategy_request(
    parsed: Dict[str, Any]
) -> Optional[str]:

    requested = normalize_text(
        parsed.get(
            "requested_strategy"
        )
    )

    if not requested:
        return None

    if requested in UNSUPPORTED_STRATEGIES:

        raise ValueError(
            f"The requested strategy "
            f"'{parsed.get('requested_strategy')}' "
            f"is not supported by the current "
            f"QuantForge engine."
        )

    if requested in SUPPORTED_STRATEGIES:
        return requested

    return None


# ============================================================
# VALIDATE RESEARCH REQUEST
# ============================================================

@app.post("/validate-research")
def validate_research(
    request: ResearchRequest
):

    # --------------------------------------------------------
    # TIMEFRAME
    # --------------------------------------------------------

    if request.timeframe not in ALLOWED_TIMEFRAMES:

        return {
            "request_valid": False,
            "reason": "Invalid timeframe selected.",
            "yahoo_verification": {
                "verified": False,
                "message": "Market-data lookup skipped.",
            },
        }

    # --------------------------------------------------------
    # CAPITAL
    # --------------------------------------------------------

    if request.starting_capital < MIN_STARTING_CAPITAL:

        return {
            "request_valid": False,
            "reason": (
                f"Starting capital must be at least "
                f"{MIN_STARTING_CAPITAL}."
            ),
            "yahoo_verification": {
                "verified": False,
                "message": "Market-data lookup skipped.",
            },
        }

    if request.starting_capital > MAX_STARTING_CAPITAL:

        return {
            "request_valid": False,
            "reason": (
                f"Starting capital cannot exceed "
                f"{MAX_STARTING_CAPITAL:,}."
            ),
            "yahoo_verification": {
                "verified": False,
                "message": "Market-data lookup skipped.",
            },
        }

    # --------------------------------------------------------
    # EXPERIMENTS
    # --------------------------------------------------------

    if (
        request.max_experiments < MIN_EXPERIMENTS
        or
        request.max_experiments > MAX_EXPERIMENTS
    ):

        return {
            "request_valid": False,
            "reason": (
                "Maximum experiments must be "
                "between 1 and 20."
            ),
            "yahoo_verification": {
                "verified": False,
                "message": "Market-data lookup skipped.",
            },
        }

    # --------------------------------------------------------
    # UNSUPPORTED STRATEGY
    # --------------------------------------------------------

    unsupported = (
        detect_explicit_unsupported_strategy(
            request.request
        )
    )

    if unsupported:

        return {
            "request_valid": False,
            "reason": (
                f"'{unsupported}' is not supported "
                "by the current QuantForge engine. "
                "Supported strategy families are "
                "EMA_Crossover, RSI_Mean_Reversion "
                "and Supertrend."
            ),
            "yahoo_verification": {
                "verified": False,
                "message": "Market-data lookup skipped.",
            },
        }

    # --------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------

    try:

        parsed = ask_ollama(
            request.request
        )

    except requests.exceptions.ConnectionError:

        return {
            "request_valid": False,
            "reason": (
                "QuantForge could not connect "
                "to Ollama at 127.0.0.1:11434."
            ),
            "yahoo_verification": {
                "verified": False,
                "message": "Request parsing unavailable.",
            },
        }

    except Exception as error:

        return {
            "request_valid": False,
            "reason": (
                f"Request understanding failed: {error}"
            ),
            "yahoo_verification": {
                "verified": False,
                "message": "Request understanding failed.",
            },
        }

    # --------------------------------------------------------
    # FINANCIAL INTENT
    # --------------------------------------------------------

    if parsed.get("request_valid") is not True:

        return {
            **parsed,
            "request_valid": False,
            "reason": parsed.get(
                "reason",
                "The request is not a financial "
                "strategy research request.",
            ),
            "yahoo_verification": {
                "verified": False,
                "message": "Market-data lookup skipped.",
            },
            "timeframe": request.timeframe,
            "starting_capital": request.starting_capital,
            "max_experiments": request.max_experiments,
        }

    # --------------------------------------------------------
    # STRATEGY
    # --------------------------------------------------------

    try:

        requested_strategy = (
            check_strategy_request(
                parsed
            )
        )

    except ValueError as error:

        return {
            **parsed,
            "request_valid": False,
            "reason": str(error),
            "yahoo_verification": {
                "verified": False,
                "message": "Market-data lookup skipped.",
            },
        }

    # --------------------------------------------------------
    # INSTRUMENT
    # --------------------------------------------------------

    instrument = resolve_instrument(
        parsed
    )

    if not instrument:

        return {
            **parsed,
            "request_valid": False,
            "reason": (
                "The requested asset could not be "
                "resolved to a supported QuantForge "
                "Forex instrument. Currently supported: "
                "EUR/USD, GBP/USD, USD/JPY, USD/CHF, "
                "AUD/USD, USD/CAD and NZD/USD."
            ),
            "yahoo_verification": {
                "verified": False,
                "message": "No supported instrument was resolved.",
            },
            "timeframe": request.timeframe,
            "starting_capital": request.starting_capital,
            "max_experiments": request.max_experiments,
        }

    # --------------------------------------------------------
    # YAHOO
    # --------------------------------------------------------

    yahoo_result = verify_yahoo_symbol(
        instrument
    )

    if not yahoo_result.get("verified"):

        return {
            **parsed,
            "request_valid": False,
            "reason": (
                "The financial request is valid, "
                "but Yahoo Finance verification failed: "
                +
                yahoo_result.get(
                    "message",
                    "Unknown error."
                )
            ),
            "instrument": instrument,
            "yahoo_verification": yahoo_result,
            "timeframe": request.timeframe,
            "starting_capital": request.starting_capital,
            "max_experiments": request.max_experiments,
        }

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return {
        **parsed,
        "request_valid": True,
        "reason": parsed.get(
            "reason",
            "Valid QuantForge research request."
        ),
        "research_objective": (
            parsed.get(
                "research_objective"
            )
            or request.request
        ),
        "instrument": instrument,
        "asset_name": SUPPORTED_FOREX[
            instrument
        ]["name"],
        "yahoo_symbol": SUPPORTED_FOREX[
            instrument
        ]["yahoo"],
        "timeframe": request.timeframe,
        "starting_capital": request.starting_capital,
        "max_experiments": request.max_experiments,
        "research_run_id": generate_research_run_id(),
        "requested_strategy": requested_strategy,
        "yahoo_verification": yahoo_result,
    }


# ============================================================
# BACKGROUND N8N EXECUTION
# ============================================================

def send_progress(
    research_run_id: str,
    status: str,
    stage: str,
    message: str,
    **extra
):

    payload = {
        "research_run_id": research_run_id,
        "status": status,
        "stage": stage,
        "message": message,
        **extra,
    }

    try:

        requests.post(
            f"{PROGRESS_SERVER_URL}/research-progress",
            json=payload,
            timeout=5,
        )

    except Exception:
        # Progress reporting must never kill research.
        pass


def run_n8n_background(
    payload: Dict[str, Any]
):

    research_run_id = (
        payload["research_run_id"]
    )

    try:

        send_progress(
            research_run_id,
            "RUNNING",
            "RESEARCH_STARTED",
            "QuantForge research workflow started.",
            experiment=1,
            max_experiments=payload[
                "max_experiments"
            ],
        )

        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=N8N_TIMEOUT_SECONDS,
        )

        if not response.ok:

            send_progress(
                research_run_id,
                "FAILED",
                "ERROR",
                (
                    f"n8n returned HTTP "
                    f"{response.status_code}."
                ),
            )

            return

        try:
            result = response.json()
        except Exception:
            result = {
                "raw_response": response.text[:5000]
            }

        # If the workflow's Progress — Completed node
        # has already reported completion, this update
        # simply refreshes the final state.
        send_progress(
            research_run_id,
            "COMPLETED",
            "COMPLETED",
            "QuantForge research completed.",
            result=result,
        )

    except requests.exceptions.Timeout:

        send_progress(
            research_run_id,
            "FAILED",
            "ERROR",
            "QuantForge research exceeded the 60-minute execution limit.",
        )

    except requests.exceptions.ConnectionError:

        send_progress(
            research_run_id,
            "FAILED",
            "ERROR",
            (
                "Could not connect to the n8n production "
                "webhook."
            ),
        )

    except Exception as error:

        send_progress(
            research_run_id,
            "FAILED",
            "ERROR",
            f"Research workflow failed: {error}",
        )


# ============================================================
# RUN RESEARCH
# ============================================================

@app.post("/run-research")
def run_research(
    request: ResearchRequest
):

    # --------------------------------------------------------
    # VALIDATE FIRST
    # --------------------------------------------------------

    validation = validate_research(
        request
    )

    if not validation.get(
        "request_valid"
    ):

        return validation

    research_run_id = (
        validation["research_run_id"]
    )

    # --------------------------------------------------------
    # N8N PAYLOAD
    # --------------------------------------------------------

    payload = {
        "research_objective": validation[
            "research_objective"
        ],

        "instrument": validation[
            "instrument"
        ],

        "timeframe": validation[
            "timeframe"
        ],

        "starting_capital": validation[
            "starting_capital"
        ],

        "max_experiments": validation[
            "max_experiments"
        ],

        "experiment": 1,

        "research_run_id": research_run_id,
    }

    # --------------------------------------------------------
    # INITIAL PROGRESS STATE
    # --------------------------------------------------------

    send_progress(
        research_run_id,
        "RUNNING",
        "STARTING",
        "Research request accepted. Starting n8n workflow.",
        experiment=1,
        max_experiments=validation[
            "max_experiments"
        ],
        instrument=validation[
            "instrument"
        ],
        timeframe=validation[
            "timeframe"
        ],
        starting_capital=validation[
            "starting_capital"
        ],
        research_objective=validation[
            "research_objective"
        ],
    )

    # --------------------------------------------------------
    # START N8N WITHOUT BLOCKING THE DASHBOARD
    # --------------------------------------------------------

    worker = threading.Thread(
        target=run_n8n_background,
        args=(payload,),
        daemon=True,
    )

    worker.start()

    # --------------------------------------------------------
    # RETURN IMMEDIATELY
    # --------------------------------------------------------

    return {
        "request_valid": True,
        "status": "RUNNING",
        "research_run_id": research_run_id,
        "message": (
            "QuantForge research started."
        ),
        "research_objective": validation[
            "research_objective"
        ],
        "instrument": validation[
            "instrument"
        ],
        "asset_name": validation[
            "asset_name"
        ],
        "timeframe": validation[
            "timeframe"
        ],
        "starting_capital": validation[
            "starting_capital"
        ],
        "max_experiments": validation[
            "max_experiments"
        ],
        "request_validation": validation,
    }


# ============================================================
# LIVE RESEARCH STATUS
# ============================================================

@app.get(
    "/research-status/{research_run_id}"
)
def research_status(
    research_run_id: str
):

    try:

        response = requests.get(
            (
                f"{PROGRESS_SERVER_URL}"
                f"/research-status/"
                f"{research_run_id}"
            ),
            timeout=5,
        )

        if response.status_code == 404:

            return {
                "research_run_id": research_run_id,
                "status": "STARTING",
                "stage": "STARTING",
                "message": (
                    "Waiting for the first progress update..."
                ),
            }

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError:

        raise HTTPException(
            status_code=503,
            detail=(
                "QuantForge Progress Server is not "
                "running at 127.0.0.1:8002."
            ),
        )

    except Exception as error:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Could not retrieve research status: "
                f"{error}"
            ),
        )


# ============================================================
# PDF DOWNLOAD
# ============================================================

@app.get(
    "/download-report/{filename}"
)
def download_report(
    filename: str
):

    safe_filename = Path(
        filename
    ).name

    report_path = (
        REPORT_DIRECTORY /
        safe_filename
    )

    if not report_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Research report was not found.",
        )

    try:

        report_path.resolve().relative_to(
            REPORT_DIRECTORY.resolve()
        )

    except ValueError:

        raise HTTPException(
            status_code=403,
            detail="Invalid report path.",
        )

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=safe_filename,
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    ollama_available = False
    progress_available = False

    try:

        response = requests.get(
            "http://127.0.0.1:11434/api/tags",
            timeout=3,
        )

        ollama_available = response.ok

    except Exception:
        pass

    try:

        response = requests.get(
            f"{PROGRESS_SERVER_URL}/health",
            timeout=3,
        )

        progress_available = response.ok

    except Exception:
        pass

    return {
        "status": "QuantForge API is running",
        "ollama": ollama_available,
        "progress_server": progress_available,
        "n8n_webhook": N8N_WEBHOOK_URL,
        "supported_instruments": list(
            SUPPORTED_FOREX.keys()
        ),
        "supported_timeframes": sorted(
            ALLOWED_TIMEFRAMES
        ),
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return r"""
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>QuantForge — Research Dashboard</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Inter,
        Segoe UI,
        Arial,
        sans-serif;

    background: #0b0f19;

    color: #e5e7eb;
}

.container {

    width: min(1250px, 94%);

    margin: 0 auto;
}

.header {

    padding: 28px 0 18px;

    border-bottom:
        1px solid #1f2937;
}

.brand {

    display: flex;

    align-items: center;

    gap: 12px;
}

.logo {

    width: 42px;
    height: 42px;

    border-radius: 12px;

    display: flex;

    align-items: center;
    justify-content: center;

    background: #111827;

    border:
        1px solid #374151;

    font-weight: 800;
}

h1 {

    margin: 0;

    font-size: 24px;
}

.subtitle {

    margin-top: 5px;

    color: #9ca3af;

    font-size: 14px;
}

.grid {

    display: grid;

    grid-template-columns:
        360px 1fr;

    gap: 22px;

    margin-top: 24px;
}

.card {

    background: #111827;

    border:
        1px solid #1f2937;

    border-radius: 16px;

    padding: 22px;

    box-shadow:
        0 12px 40px rgba(0,0,0,.18);
}

.card h2 {

    margin:
        0 0 18px;

    font-size: 17px;
}

label {

    display: block;

    margin:
        16px 0 7px;

    font-size: 13px;

    font-weight: 700;

    color: #d1d5db;
}

textarea,
select,
input {

    width: 100%;

    background: #0b0f19;

    border:
        1px solid #374151;

    color: #f9fafb;

    border-radius: 10px;

    padding: 12px;

    outline: none;

    font-size: 14px;
}

textarea {

    min-height: 130px;

    resize: vertical;
}

textarea:focus,
select:focus,
input:focus {

    border-color: #60a5fa;
}

button {

    width: 100%;

    margin-top: 20px;

    padding: 13px;

    border: 0;

    border-radius: 10px;

    background: #2563eb;

    color: white;

    font-weight: 700;

    cursor: pointer;
}

button:hover {

    background: #1d4ed8;
}

button:disabled {

    opacity: .55;

    cursor: wait;
}

.status {

    display: none;

    margin-top: 16px;

    padding: 12px;

    border-radius: 10px;

    font-size: 13px;

    line-height: 1.5;
}

.status.show {
    display: block;
}

.status.info {

    background: #172554;

    border:
        1px solid #1d4ed8;
}

.status.error {

    background: #450a0a;

    border:
        1px solid #991b1b;
}

.status.success {

    background: #052e16;

    border:
        1px solid #166534;
}

.progress {

    display: none;

    margin-top: 18px;
}

.progress.show {
    display: block;
}

.progress-header {

    display: flex;

    justify-content:
        space-between;

    align-items: center;

    margin-bottom: 10px;
}

.progress-title {

    font-size: 14px;

    font-weight: 800;
}

.progress-percent {

    font-size: 13px;

    color: #60a5fa;
}

.progress-bar {

    width: 100%;

    height: 8px;

    background: #1f2937;

    border-radius: 999px;

    overflow: hidden;
}

.progress-fill {

    height: 100%;

    width: 0%;

    background: #2563eb;

    transition:
        width .4s ease;
}

.stage-list {

    margin-top: 18px;

    display: grid;

    gap: 8px;
}

.stage {

    display: flex;

    align-items: center;

    gap: 10px;

    padding: 9px;

    border-radius: 8px;

    background: #0b0f19;

    border:
        1px solid #1f2937;

    font-size: 12px;

    color: #6b7280;
}

.stage.active {

    color: #93c5fd;

    border-color: #1d4ed8;

    background: #172554;
}

.stage.done {

    color: #86efac;

    border-color: #166534;
}

.stage.failed {

    color: #fca5a5;

    border-color: #991b1b;
}

.stage-dot {

    width: 9px;
    height: 9px;

    border-radius: 50%;

    background: #374151;

    flex-shrink: 0;
}

.stage.active .stage-dot {

    background: #60a5fa;
}

.stage.done .stage-dot {

    background: #22c55e;
}

.stage.failed .stage-dot {

    background: #ef4444;
}

.live-info {

    display: none;

    margin-top: 16px;

    padding: 14px;

    border-radius: 10px;

    background: #0b0f19;

    border:
        1px solid #1f2937;
}

.live-info.show {
    display: block;
}

.live-row {

    display: flex;

    justify-content:
        space-between;

    gap: 15px;

    padding: 6px 0;

    font-size: 12px;
}

.live-label {

    color: #9ca3af;
}

.live-value {

    font-weight: 700;

    text-align: right;
}

.results {

    display: none;
}

.results.show {
    display: block;
}

.metrics {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 12px;
}

.metric {

    background: #0b0f19;

    border:
        1px solid #1f2937;

    border-radius: 12px;

    padding: 15px;
}

.metric-label {

    color: #9ca3af;

    font-size: 12px;
}

.metric-value {

    margin-top: 5px;

    font-size: 20px;

    font-weight: 800;
}

.meta {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 12px;

    margin-bottom: 18px;
}

.meta-item {

    background: #0b0f19;

    border:
        1px solid #1f2937;

    padding: 13px;

    border-radius: 10px;
}

.meta-label {

    color: #9ca3af;

    font-size: 11px;
}

.meta-value {

    margin-top: 4px;

    font-weight: 700;
}

.badge {

    display: inline-block;

    padding: 5px 9px;

    border-radius: 999px;

    font-size: 11px;

    font-weight: 800;
}

.badge.cool {

    background: #172554;

    color: #93c5fd;
}

.badge.cold {

    background: #1f2937;

    color: #d1d5db;
}

.badge.hot {

    background: #451a03;

    color: #fdba74;
}

.badge.pass {

    background: #052e16;

    color: #86efac;
}

.badge.fail {

    background: #450a0a;

    color: #fca5a5;
}

.objective {

    padding: 14px;

    background: #0b0f19;

    border:
        1px solid #1f2937;

    border-radius: 10px;

    margin-bottom: 18px;

    line-height: 1.5;
}

.table-wrap {

    overflow-x: auto;

    margin-top: 18px;
}

table {

    width: 100%;

    border-collapse:
        collapse;

    font-size: 12px;
}

th,
td {

    text-align: left;

    padding: 10px;

    border-bottom:
        1px solid #1f2937;

    white-space: nowrap;
}

th {

    color: #9ca3af;

    font-size: 11px;
}

.download {

    display: inline-block;

    margin-top: 18px;

    padding: 11px 16px;

    border-radius: 9px;

    background: #374151;

    color: white;

    text-decoration: none;

    font-weight: 700;
}

.download:hover {

    background: #4b5563;
}

.empty {

    color: #6b7280;

    text-align: center;

    padding: 70px 20px;
}

@media (max-width: 900px) {

    .grid {

        grid-template-columns: 1fr;
    }

    .metrics {

        grid-template-columns:
            repeat(2, 1fr);
    }

    .meta {

        grid-template-columns: 1fr;
    }
}

</style>

</head>

<body>

<div class="container">

<div class="header">

<div class="brand">

<div class="logo">QF</div>

<div>

<h1>QuantForge</h1>

<div class="subtitle">
Agentic Forex Strategy Research & Validation
</div>

</div>

</div>

</div>


<div class="grid">


<!-- ================================================= -->
<!-- RESEARCH FORM -->
<!-- ================================================= -->

<div class="card">

<h2>Start Research</h2>

<label for="request">
Research objective
</label>

<textarea
    id="request"
    placeholder="Example: Find a trading strategy for EUR/USD that improves risk-adjusted performance while keeping drawdown reasonably low."
></textarea>

<label for="timeframe">
Timeframe
</label>

<select id="timeframe">

<option value="15m">
15 minutes
</option>

<option value="30m">
30 minutes
</option>

<option value="1h">
1 hour
</option>

<option value="4h">
4 hours
</option>

<option value="1d">
1 day
</option>

</select>

<label for="capital">
Starting capital
</label>

<input
    id="capital"
    type="number"
    min="1000"
    max="100000000"
    step="100"
    value="10000"
>

<label for="experiments">
Maximum experiments
</label>

<input
    id="experiments"
    type="number"
    min="1"
    max="20"
    step="1"
    value="5"
>

<button
    id="runButton"
    onclick="runResearch()"
>
Run QuantForge Research
</button>


<div
    id="status"
    class="status"
></div>


<!-- ================================================= -->
<!-- LIVE PROGRESS -->
<!-- ================================================= -->

<div
    id="progress"
    class="progress"
>

<div class="progress-header">

<div
    id="progressTitle"
    class="progress-title"
>
Preparing research...
</div>

<div
    id="progressPercent"
    class="progress-percent"
>
0%
</div>

</div>


<div class="progress-bar">

<div
    id="progressFill"
    class="progress-fill"
></div>

</div>


<div
    id="stageList"
    class="stage-list"
>

<div
    class="stage"
    data-stage="RESEARCHER"
>
<span class="stage-dot"></span>
Researcher
</div>

<div
    class="stage"
    data-stage="BACKTEST"
>
<span class="stage-dot"></span>
Backtest
</div>

<div
    class="stage"
    data-stage="HEAT"
>
<span class="stage-dot"></span>
Heat Analysis
</div>

<div
    class="stage"
    data-stage="CRITIC"
>
<span class="stage-dot"></span>
Critic / Experimental Planner
</div>

<div
    class="stage"
    data-stage="VALIDATION"
>
<span class="stage-dot"></span>
Validation
</div>

<div
    class="stage"
    data-stage="REPORT"
>
<span class="stage-dot"></span>
Report Generation
</div>

<div
    class="stage"
    data-stage="COMPLETED"
>
<span class="stage-dot"></span>
Completed
</div>

</div>


<div
    id="liveInfo"
    class="live-info"
>

<div class="live-row">

<span class="live-label">
Experiment
</span>

<span
    id="liveExperiment"
    class="live-value"
>
—
</span>

</div>

<div class="live-row">

<span class="live-label">
Current Stage
</span>

<span
    id="liveStage"
    class="live-value"
>
—
</span>

</div>

<div class="live-row">

<span class="live-label">
Strategy
</span>

<span
    id="liveStrategy"
    class="live-value"
>
—
</span>

</div>

<div class="live-row">

<span class="live-label">
Return
</span>

<span
    id="liveReturn"
    class="live-value"
>
—
</span>

</div>

<div class="live-row">

<span class="live-label">
Sharpe
</span>

<span
    id="liveSharpe"
    class="live-value"
>
—
</span>

</div>

<div class="live-row">

<span class="live-label">
Drawdown
</span>

<span
    id="liveDrawdown"
    class="live-value"
>
—
</span>

</div>

<div class="live-row">

<span class="live-label">
Heat
</span>

<span
    id="liveHeat"
    class="live-value"
>
—
</span>

</div>

</div>

</div>

</div>


<!-- ================================================= -->
<!-- RESULTS -->
<!-- ================================================= -->

<div
    id="results"
    class="card results"
>

<h2>Research Results</h2>

<div
    id="objective"
    class="objective"
></div>

<div
    id="meta"
    class="meta"
></div>

<div
    id="metrics"
    class="metrics"
></div>

<div
    id="validation"
    style="margin-top:18px;"
></div>

<a
    id="download"
    class="download"
    href="#"
    target="_blank"
    style="display:none;"
>
Download Research PDF
</a>

<div class="table-wrap">

<h2 style="margin-top:25px;">
Experiment History
</h2>

<table>

<thead>

<tr>

<th>Exp</th>
<th>Strategy</th>
<th>Parameters</th>
<th>Return</th>
<th>Trades</th>
<th>Win Rate</th>
<th>PF</th>
<th>Sharpe</th>
<th>Drawdown</th>
<th>Heat</th>

</tr>

</thead>

<tbody
    id="historyBody"
></tbody>

</table>

</div>

</div>


<div
    id="emptyResults"
    class="card empty"
>

Run a research request to see
the autonomous experiment results.

</div>


</div>

</div>


<script>


// ==========================================================
// GLOBAL STATE
// ==========================================================

let pollingTimer = null;

let currentResearchRunId = null;

let currentMaxExperiments = 1;


// ==========================================================
// HELPERS
// ==========================================================

function escapeHtml(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return "";
    }

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function formatNumber(
    value,
    decimals = 2
) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {

        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {

        return "—";
    }

    return number.toFixed(decimals);
}


function formatPercent(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return "—";
    }

    return formatNumber(value) + "%";
}


function formatParameters(config) {

    if (
        !config ||
        !config.parameters
    ) {

        return "—";
    }

    return Object.entries(
        config.parameters
    )
    .map(
        ([key, value]) =>
            `${escapeHtml(key)}=${escapeHtml(value)}`
    )
    .join(", ");
}


function heatBadge(score, label) {

    let cls = "cool";

    const normalized =
        String(label || "").toUpperCase();

    if (normalized === "COLD") {

        cls = "cold";
    }

    if (
        normalized === "HOT" ||
        normalized === "VERY HOT"
    ) {

        cls = "hot";
    }

    return `
        <span class="badge ${cls}">
            ${escapeHtml(score)} / 4
            ${escapeHtml(label || "")}
        </span>
    `;
}


function showStatus(
    message,
    type
) {

    const status =
        document.getElementById("status");

    status.className =
        "status show " + type;

    status.textContent =
        message;
}


function setProgressVisible(visible) {

    const progress =
        document.getElementById("progress");

    progress.classList.toggle(
        "show",
        visible
    );
}


function stageGroup(stage) {

    const s =
        String(stage || "").toUpperCase();

    if (
        s.includes("RESEARCHER")
    ) {
        return "RESEARCHER";
    }

    if (
        s.includes("BACKTEST")
    ) {
        return "BACKTEST";
    }

    if (
        s.includes("HEAT")
    ) {
        return "HEAT";
    }

    if (
        s.includes("CRITIC")
    ) {
        return "CRITIC";
    }

    if (
        s.includes("VALIDATION")
    ) {
        return "VALIDATION";
    }

    if (
        s.includes("REPORT")
    ) {
        return "REPORT";
    }

    if (
        s.includes("COMPLETED")
    ) {
        return "COMPLETED";
    }

    return null;
}


const stageOrder = [
    "RESEARCHER",
    "BACKTEST",
    "HEAT",
    "CRITIC",
    "VALIDATION",
    "REPORT",
    "COMPLETED"
];


function updateStageUI(
    stage,
    status
) {

    const current =
        stageGroup(stage);

    const elements =
        document.querySelectorAll(
            ".stage"
        );

    let currentIndex =
        stageOrder.indexOf(current);

    if (currentIndex < 0) {
        currentIndex = 0;
    }

    elements.forEach(
        element => {

            const name =
                element.dataset.stage;

            const index =
                stageOrder.indexOf(name);

            element.classList.remove(
                "active",
                "done",
                "failed"
            );

            if (
                status === "FAILED"
            ) {

                if (
                    name === current
                ) {

                    element.classList.add(
                        "failed"
                    );
                }

                return;
            }

            if (
                index < currentIndex
            ) {

                element.classList.add(
                    "done"
                );

            }
            else if (
                index === currentIndex
            ) {

                element.classList.add(
                    "active"
                );
            }

            if (
                status === "COMPLETED"
                &&
                name === "COMPLETED"
            ) {

                element.classList.add(
                    "done"
                );
            }

        }
    );
}


function calculateProgress(data) {

    const status =
        String(
            data.status || ""
        ).toUpperCase();

    if (status === "COMPLETED") {
        return 100;
    }

    if (status === "FAILED") {
        return 100;
    }

    const experiment =
        Number(data.experiment || 1);

    const maxExperiments =
        Number(
            data.max_experiments ||
            currentMaxExperiments ||
            1
        );

    const stage =
        stageGroup(data.stage);

    const stageIndex =
        Math.max(
            0,
            stageOrder.indexOf(stage)
        );

    const stagesPerExperiment = 4;

    const experimentFraction =
        Math.max(
            0,
            Math.min(
                1,
                (experiment - 1) /
                maxExperiments
            )
        );

    const stageFraction =
        Math.max(
            0,
            Math.min(
                1,
                (stageIndex + 1) /
                stagesPerExperiment
            )
        );

    let value =
        (
            experimentFraction +
            stageFraction /
            maxExperiments
        ) * 100;

    return Math.max(
        2,
        Math.min(96, value)
    );
}


// ==========================================================
// UPDATE LIVE UI
// ==========================================================

function updateLiveProgress(data) {

    const progress =
        document.getElementById(
            "progress"
        );

    progress.classList.add(
        "show"
    );

    const stage =
        data.stage ||
        "STARTING";

    const message =
        data.message ||
        stage;

    document.getElementById(
        "progressTitle"
    ).textContent =
        message;

    const percent =
        Math.round(
            calculateProgress(data)
        );

    document.getElementById(
        "progressPercent"
    ).textContent =
        percent + "%";

    document.getElementById(
        "progressFill"
    ).style.width =
        percent + "%";

    updateStageUI(
        stage,
        data.status
    );


    document.getElementById(
        "liveInfo"
    ).classList.add(
        "show"
    );


    const experiment =
        data.experiment;

    const maxExperiments =
        data.max_experiments ||
        currentMaxExperiments;

    document.getElementById(
        "liveExperiment"
    ).textContent =
        experiment
            ? `${experiment} / ${maxExperiments}`
            : "—";


    document.getElementById(
        "liveStage"
    ).textContent =
        stage;


    const strategy =
        data.strategy ||
        (
            data.strategy_config &&
            data.strategy_config.strategy
        );

    document.getElementById(
        "liveStrategy"
    ).textContent =
        strategy || "—";


    const metrics =
        data.metrics || {};


    document.getElementById(
        "liveReturn"
    ).textContent =
        formatPercent(
            metrics.return_percent ??
            data.return_percent
        );


    document.getElementById(
        "liveSharpe"
    ).textContent =
        formatNumber(
            metrics.sharpe_ratio ??
            data.sharpe_ratio
        );


    document.getElementById(
        "liveDrawdown"
    ).textContent =
        formatPercent(
            metrics.max_drawdown_percent ??
            data.max_drawdown_percent
        );


    const heat =
        data.heat || {};

    document.getElementById(
        "liveHeat"
    ).innerHTML =
        heat.score !== undefined ||
        heat.heat_score !== undefined

            ? heatBadge(
                heat.score ??
                heat.heat_score,
                heat.label ??
                heat.heat_label
            )

            : "—";
}


// ==========================================================
// POLLING
// ==========================================================

async function pollResearchStatus() {

    if (!currentResearchRunId) {
        return;
    }

    try {

        const response =
            await fetch(
                "/research-status/" +
                encodeURIComponent(
                    currentResearchRunId
                ),
                {
                    cache: "no-store"
                }
            );

        if (!response.ok) {

            throw new Error(
                "Status request failed: HTTP " +
                response.status
            );
        }

        const data =
            await response.json();

        updateLiveProgress(
            data
        );


        if (
            data.status === "COMPLETED"
        ) {

            stopPolling();

            showStatus(
                "Research completed successfully.",
                "success"
            );

            document.getElementById(
                "runButton"
            ).disabled = false;

            document.getElementById(
                "runButton"
            ).textContent =
                "Run QuantForge Research";


            const result =
                data.result ||
                data.dashboard ||
                data;

            renderResults(
                result
            );

            return;
        }


        if (
            data.status === "FAILED"
        ) {

            stopPolling();

            showStatus(
                data.message ||
                "QuantForge research failed.",
                "error"
            );

            document.getElementById(
                "runButton"
            ).disabled = false;

            document.getElementById(
                "runButton"
            ).textContent =
                "Run QuantForge Research";

            return;
        }


        pollingTimer =
            setTimeout(
                pollResearchStatus,
                1500
            );

    }
    catch (error) {

        console.error(
            "Progress polling error:",
            error
        );

        pollingTimer =
            setTimeout(
                pollResearchStatus,
                2500
            );
    }
}


function stopPolling() {

    if (pollingTimer) {

        clearTimeout(
            pollingTimer
        );

        pollingTimer = null;
    }
}


// ==========================================================
// RUN RESEARCH
// ==========================================================

async function runResearch() {

    stopPolling();

    const button =
        document.getElementById(
            "runButton"
        );

    const request =
        document.getElementById(
            "request"
        ).value.trim();

    const timeframe =
        document.getElementById(
            "timeframe"
        ).value;

    const capital =
        Number(
            document.getElementById(
                "capital"
            ).value
        );

    const experiments =
        Number(
            document.getElementById(
                "experiments"
            ).value
        );


    // ------------------------------------------------------
    // CLIENT VALIDATION
    // ------------------------------------------------------

    if (!request) {

        showStatus(
            "Enter a research objective first.",
            "error"
        );

        return;
    }


    if (
        !Number.isFinite(capital) ||
        capital < 1000 ||
        capital > 100000000
    ) {

        showStatus(
            "Starting capital must be between ₹1,000 and ₹100,000,000.",
            "error"
        );

        return;
    }


    if (
        !Number.isInteger(experiments) ||
        experiments < 1 ||
        experiments > 20
    ) {

        showStatus(
            "Maximum experiments must be between 1 and 20.",
            "error"
        );

        return;
    }


    currentMaxExperiments =
        experiments;


    button.disabled = true;

    button.textContent =
        "Starting research...";


    setProgressVisible(
        true
    );


    document.getElementById(
        "progressTitle"
    ).textContent =
        "Validating research request...";


    document.getElementById(
        "progressPercent"
    ).textContent =
        "0%";


    document.getElementById(
        "progressFill"
    ).style.width =
        "0%";


    document.getElementById(
        "liveInfo"
    ).classList.remove(
        "show"
    );


    showStatus(
        "QuantForge is validating the request and starting the autonomous research loop.",
        "info"
    );


    try {

        const response =
            await fetch(
                "/run-research",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            request:
                                request,

                            timeframe:
                                timeframe,

                            starting_capital:
                                capital,

                            max_experiments:
                                experiments
                        })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Research request failed."
            );
        }


        if (
            data.request_valid === false
        ) {

            showStatus(
                data.reason ||
                "The research request was rejected.",
                "error"
            );

            button.disabled = false;

            button.textContent =
                "Run QuantForge Research";

            setProgressVisible(
                false
            );

            return;
        }


        // --------------------------------------------------
        // IMPORTANT:
        // /run-research NOW RETURNS IMMEDIATELY.
        // --------------------------------------------------

        currentResearchRunId =
            data.research_run_id;


        showStatus(
            "Research is running. Live progress is being updated below.",
            "info"
        );


        document.getElementById(
            "progressTitle"
        ).textContent =
            "Research started...";


        document.getElementById(
            "progressPercent"
        ).textContent =
            "1%";


        document.getElementById(
            "progressFill"
        ).style.width =
            "1%";


        button.textContent =
            "Research running...";


        pollResearchStatus();

    }
    catch (error) {

        console.error(
            error
        );

        showStatus(
            error.message ||
            "Could not start research.",
            "error"
        );

        button.disabled = false;

        button.textContent =
            "Run QuantForge Research";
    }
}


// ==========================================================
// RENDER FINAL RESULTS
// ==========================================================

function renderResults(result) {

    const dashboard =
        result.dashboard ||
        {};


    const results =
        document.getElementById(
            "results"
        );

    const empty =
        document.getElementById(
            "emptyResults"
        );


    empty.style.display =
        "none";

    results.classList.add(
        "show"
    );


    // ------------------------------------------------------
    // OBJECTIVE
    // ------------------------------------------------------

    document.getElementById(
        "objective"
    ).innerHTML = `

        <strong>
            Research Objective
        </strong>

        <div style="margin-top:7px;">

            ${escapeHtml(
                dashboard.research_objective ||
                result.research_objective ||
                "Not provided"
            )}

        </div>
    `;


    // ------------------------------------------------------
    // META
    // ------------------------------------------------------

    document.getElementById(
        "meta"
    ).innerHTML = `

        <div class="meta-item">

            <div class="meta-label">
                Instrument
            </div>

            <div class="meta-value">
                ${escapeHtml(
                    dashboard.instrument ||
                    result.instrument ||
                    "—"
                )}
            </div>

        </div>


        <div class="meta-item">

            <div class="meta-label">
                Timeframe
            </div>

            <div class="meta-value">
                ${escapeHtml(
                    dashboard.timeframe ||
                    result.timeframe ||
                    "—"
                )}
            </div>

        </div>


        <div class="meta-item">

            <div class="meta-label">
                Selected Strategy
            </div>

            <div class="meta-value">
                ${escapeHtml(
                    dashboard.strategy ||
                    result.strategy ||
                    "—"
                )}
            </div>

        </div>

    `;


    // ------------------------------------------------------
    // METRICS
    // ------------------------------------------------------

    document.getElementById(
        "metrics"
    ).innerHTML = `

        <div class="metric">

            <div class="metric-label">
                Return
            </div>

            <div class="metric-value">
                ${formatPercent(
                    dashboard.return_percent
                )}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Profit Factor
            </div>

            <div class="metric-value">
                ${formatNumber(
                    dashboard.profit_factor
                )}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Sharpe Ratio
            </div>

            <div class="metric-value">
                ${formatNumber(
                    dashboard.sharpe_ratio
                )}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Max Drawdown
            </div>

            <div class="metric-value">
                ${formatPercent(
                    dashboard.max_drawdown_percent
                )}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Trades
            </div>

            <div class="metric-value">
                ${escapeHtml(
                    dashboard.trades ??
                    "—"
                )}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Win Rate
            </div>

            <div class="metric-value">
                ${formatPercent(
                    dashboard.win_rate
                )}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Heat
            </div>

            <div class="metric-value">

                ${heatBadge(
                    dashboard.heat_score,
                    dashboard.heat_label
                )}

            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Best Experiment
            </div>

            <div class="metric-value">
                #${escapeHtml(
                    dashboard.best_experiment ??
                    "—"
                )}
            </div>

        </div>

    `;


    // ------------------------------------------------------
    // VALIDATION
    // ------------------------------------------------------

    const validation =
        dashboard.validation;

    const validationElement =
        document.getElementById(
            "validation"
        );

    const validationText =
        String(
            validation || ""
        ).toUpperCase();

    const validationClass =
        validationText === "PASS"
            ? "pass"
            : "fail";

    validationElement.innerHTML = `

        <span class="badge ${validationClass}">

            Validation:
            ${escapeHtml(
                validation ||
                "UNKNOWN"
            )}

        </span>

        ${
            dashboard.validation_reason
                ?
                `<div style="
                    margin-top:8px;
                    color:#9ca3af;
                    font-size:13px;
                ">
                    ${escapeHtml(
                        dashboard.validation_reason
                    )}
                </div>`
                : ""
        }

    `;


    // ------------------------------------------------------
    // PDF
    // ------------------------------------------------------

    const download =
        document.getElementById(
            "download"
        );


    const downloadUrl =
        result.download_url ||
        dashboard.download_url;


    if (downloadUrl) {

        download.href =
            downloadUrl;

        download.style.display =
            "inline-block";

    }
    else {

        download.style.display =
            "none";
    }


    // ------------------------------------------------------
    // HISTORY
    // ------------------------------------------------------

    const history =
        dashboard.experiments ||
        dashboard.history ||
        result.experiments ||
        [];


    const body =
        document.getElementById(
            "historyBody"
        );


    body.innerHTML = "";


    if (
        !Array.isArray(history) ||
        history.length === 0
    ) {

        body.innerHTML = `

            <tr>

                <td
                    colspan="10"
                    style="color:#6b7280;"
                >

                    No experiment history returned.

                </td>

            </tr>
        `;

        return;
    }


    for (
        const experiment
        of history
    ) {

        const config =
            experiment.strategy_config ||
            {};


        const row =
            document.createElement(
                "tr"
            );


        row.innerHTML = `

            <td>
                ${escapeHtml(
                    experiment.experiment ??
                    "—"
                )}
            </td>


            <td>
                ${escapeHtml(
                    experiment.strategy ||
                    config.strategy ||
                    "—"
                )}
            </td>


            <td>
                ${formatParameters(
                    config
                )}
            </td>


            <td>
                ${formatPercent(
                    experiment.return_percent
                )}
            </td>


            <td>
                ${escapeHtml(
                    experiment.trades ??
                    "—"
                )}
            </td>


            <td>
                ${formatPercent(
                    experiment.win_rate
                )}
            </td>


            <td>
                ${formatNumber(
                    experiment.profit_factor
                )}
            </td>


            <td>
                ${formatNumber(
                    experiment.sharpe_ratio
                )}
            </td>


            <td>
                ${formatPercent(
                    experiment.max_drawdown_percent
                )}
            </td>


            <td>
                ${heatBadge(
                    experiment.heat_score,
                    experiment.heat_label
                )}
            </td>

        `;


        body.appendChild(
            row
        );
    }
}

</script>

</body>

</html>
"""


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
    )