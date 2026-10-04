import json
import pandas as pd

from langchain_ollama import ChatOllama
from backtest import run_strategy


# Load market data
data = pd.read_csv(
    "../data/eurusd_h1.csv",
    index_col=0,
    parse_dates=True
)


# Create LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# Ask the AI for a strategy
prompt = """
You are the strategy researcher for QuantForge.

Research objective:
Find a simple EMA crossover strategy for EUR/USD on the H1 timeframe.

Return ONLY valid JSON.

Use exactly this format:

{
  "strategy": "EMA_Crossover",
  "fast_ema": integer,
  "slow_ema": integer
}

Rules:
- fast_ema must be between 5 and 30
- slow_ema must be between 20 and 100
- slow_ema must be greater than fast_ema
- Do not include explanations.
- Do not use markdown.
"""


response = llm.invoke(prompt)


# Convert AI response into Python dictionary
config = json.loads(response.content)


print("\n===== AI GENERATED STRATEGY =====")
print(config)


# Backtest the AI-generated strategy
result = run_strategy(
    config,
    data
)


print("\n===== BACKTEST RESULT =====")

print("Strategy:", result["strategy"])
print("Fast EMA:", result["fast_ema"])
print("Slow EMA:", result["slow_ema"])
print(
    "Return:",
    round(float(result["return_percent"]), 2),
    "%"
)
print("Trades:", result["trades"])
print(
    "Win Rate:",
    round(float(result["win_rate"]), 2),
    "%"
)
print(
    "Profit Factor:",
    round(float(result["profit_factor"]), 2)
)
print(
    "Maximum Drawdown:",
    round(float(result["max_drawdown_percent"]), 2),
    "%"
)
print(
    "Sharpe Ratio:",
    round(float(result["sharpe_ratio"]), 2)
)