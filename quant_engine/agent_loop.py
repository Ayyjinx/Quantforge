import json
import pandas as pd

from langchain_ollama import ChatOllama
from backtest import run_strategy
from validator import validate_strategy
from research_report import generate_report
# ==========================================
# Load market data
# ==========================================

data = pd.read_csv(
    "../data/eurusd_h1.csv",
    index_col=0,
    parse_dates=True
)


# ==========================================
# Create LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ==========================================
# Research objective
# ==========================================

research_objective = """
Find a simple EMA crossover strategy for EUR/USD
on the H1 timeframe.

The goal is to improve risk-adjusted performance
while keeping drawdown reasonably low.
"""


# ==========================================
# Generate initial strategy
# ==========================================

def generate_strategy():

    prompt = f"""
You are the strategy researcher for QuantForge.

Research objective:

{research_objective}

Return ONLY valid JSON.

Use exactly this format:

{{
  "strategy": "EMA_Crossover",
  "fast_ema": integer,
  "slow_ema": integer
}}

Rules:

- fast_ema must be between 5 and 30
- slow_ema must be between 20 and 100
- slow_ema must be greater than fast_ema
- Do not include explanations.
- Do not use markdown.
"""

    response = llm.invoke(prompt)

    return json.loads(response.content)


# ==========================================
# Critic
# ==========================================

def critique_strategy(
    config,
    result,
    history
):

    prompt = f"""
You are the critic and experimental planner
for QuantForge.

Research objective:

{research_objective}

Current strategy:

{json.dumps(config, indent=2)}

Current backtest result:

{json.dumps(result, indent=2)}

Previous experiments:

{json.dumps(history, indent=2)}

Analyze the current strategy AND compare it
with previous experiments.

Decide what the researcher should do next.

Return ONLY valid JSON.

Use exactly this format:

{{
  "decision": "MODIFY",
  "reason": "short explanation",
  "fast_ema": integer,
  "slow_ema": integer
}}

Rules:

- decision must be either "MODIFY" or "ACCEPT"
- If performance is weak, choose MODIFY.
- If the current strategy is better than previous
  experiments, consider keeping similar parameters.
- If modifying, you MUST change at least one EMA parameter.
- fast_ema must be between 5 and 30.
- slow_ema must be between 20 and 100.
- slow_ema must be greater than fast_ema.
- Do not repeat a strategy configuration already
  tested in the experiment history.
- Do not use markdown.
- Do not include anything outside the JSON.
"""

    response = llm.invoke(prompt)

    return json.loads(response.content)


# ==========================================
# Clean backtest result
# ==========================================

def clean_result(result):

    return {
        "strategy": str(result["strategy"]),
        "fast_ema": int(result["fast_ema"]),
        "slow_ema": int(result["slow_ema"]),
        "return_percent": float(
            round(result["return_percent"], 2)
        ),
        "trades": int(result["trades"]),
        "win_rate": float(
            round(result["win_rate"], 2)
        ),
        "profit_factor": float(
            round(result["profit_factor"], 2)
        ),
        "max_drawdown_percent": float(
            round(result["max_drawdown_percent"], 2)
        ),
        "sharpe_ratio": float(
            round(result["sharpe_ratio"], 2)
        )
    }


# ==========================================
# Find best experiment
# ==========================================

def find_best_experiment(history):

    if len(history) == 0:
        return None

    best = history[0]

    for experiment in history:

        if (
            experiment["research_score"]
            > best["research_score"]
        ):
            best = experiment

    return best


# ==========================================
# Start agent
# ==========================================

print("\n========================================")
print("        QUANTFORGE AGENT")
print("========================================")


config = generate_strategy()

history = []


# ==========================================
# Experiment loop
# ==========================================

for experiment_number in range(1, 6):

    print("\n----------------------------------------")
    print(
        "EXPERIMENT",
        experiment_number
    )
    print("----------------------------------------")

    print("\nStrategy:")
    print(
        json.dumps(
            config,
            indent=2
        )
    )

    # -------------------------------
    # Backtest
    # -------------------------------

    result = run_strategy(
        config,
        data
    )

    result = clean_result(result)

    print("\nBacktest:")
    print(
        json.dumps(
            result,
            indent=2
        )
    )
    # -------------------------------
    # Validate strategy
    # -------------------------------
    validation = validate_strategy(result)
    result["research_score"] = validation["score"]
    print("\nValidation:")
    print(
        json.dumps(
            validation,
            indent=2
        )
    )
    # -------------------------------
    # Save experiment
    # -------------------------------

    history.append(result)

    # -------------------------------
    # Find best experiment
    # -------------------------------

    best = find_best_experiment(history)

    print("\nBest Experiment So Far:")

    print(
        json.dumps(
            best,
            indent=2
        )
    )

    # -------------------------------
    # Ask AI critic
    # -------------------------------

    decision = critique_strategy(
        config,
        result,
        history
    )

    print("\nAI Decision:")

    print(
        json.dumps(
            decision,
            indent=2
        )
    )

    # -------------------------------
    # Python validation gate
    # -------------------------------

    if validation["passed"] :

        print("\nPython validation passed.")
        print("Strategy accepted by QuantForge.")
        break

    # -------------------------------
    # Modify
    # -------------------------------

    config = {
        "strategy": "EMA_Crossover",
        "fast_ema": decision["fast_ema"],
        "slow_ema": decision["slow_ema"]
    }


else:

    print("\nMaximum experiments reached.")


# ==========================================
# Final result
# ==========================================

best = find_best_experiment(history)

print("\n========================================")
print("        QUANTFORGE FINISHED")
print("========================================")

report = generate_report(
    research_objective,
    history
)

print("\n===== FINAL RESEARCH REPORT =====")

print(
    json.dumps(
        report,
        indent=2
    )
)