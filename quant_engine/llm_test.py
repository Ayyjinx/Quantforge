from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


prompt = """
You are the strategy researcher for QuantForge.

QuantForge is a research system that tests Forex trading strategies.

Research objective:
Find a simple EMA crossover strategy for EUR/USD on the H1 timeframe.

Return ONLY valid JSON.

The JSON must contain exactly these fields:

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


print("\n===== GENERATED STRATEGY =====")
print(response.content)