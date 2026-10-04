import json

from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# Results from the previous backtest
backtest_result = {
    "strategy": "EMA_Crossover",
    "fast_ema": 12,
    "slow_ema": 30,
    "return_percent": 0.64,
    "trades": 398,
    "win_rate": 28.39,
    "profit_factor": 1.01,
    "max_drawdown_percent": -8.21,
    "sharpe_ratio": 0.08
}


prompt = f"""
You are the critic and experimental planner for QuantForge.

QuantForge is an autonomous Forex strategy research system.

The system has just tested a strategy.

Here are the backtest results:

{json.dumps(backtest_result, indent=2)}

Analyze the strategy and decide what the researcher should do next.

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
- If the strategy has weak performance, choose MODIFY.
- If modifying, you MUST change at least one EMA parameter.
- If modifying, keep fast_ema between 5 and 30.
- If modifying, keep slow_ema between 20 and 100.
- slow_ema must be greater than fast_ema.
- Try a meaningfully different EMA combination rather than repeating the current parameters.
- Do not include explanations outside the JSON.
- Do not use markdown.
"""


response = llm.invoke(prompt)


print("\n===== AI CRITIC =====")
print(response.content)