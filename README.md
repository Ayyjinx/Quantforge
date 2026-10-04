# QuantForge

## A Resource-Efficient Agentic AI Framework for Autonomous Forex Strategy Research and Validation

QuantForge is an open-source, Python-based agentic AI framework designed to automate the research, backtesting, evaluation, and iterative refinement of Forex trading strategies.

The system combines a local Large Language Model (LLM) with deterministic Python-based quantitative tools and n8n workflow orchestration.

> **Important:** QuantForge is a research and strategy-validation framework. It is not designed for live trading or financial advice.

---

## 1. Project Overview

Developing and evaluating trading strategies manually can require repeated cycles of:

1. Defining a strategy hypothesis
2. Configuring strategy parameters
3. Running a historical backtest
4. Analyzing performance
5. Identifying weaknesses
6. Modifying the strategy
7. Running another experiment
8. Comparing experiments
9. Validating the final candidate

QuantForge automates this research loop using an agentic architecture.

The core principle of the system is:

> **LLM decides and interprets; Python calculates and verifies.**

The LLM is responsible for reasoning, hypothesis generation, experiment planning, and interpretation.

Python is responsible for deterministic operations such as:

- Market data processing
- Technical indicators
- Strategy execution
- Backtesting
- Performance metrics
- Heat scoring
- Validation
- Report generation

n8n coordinates the complete research workflow.

---

## 2. System Architecture

```text
                    ┌───────────────────────┐
                    │   QuantForge Dashboard│
                    │      / Input Gateway  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │         n8n            │
                    │  Workflow Orchestrator │
                    └───────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
       ┌─────────────────┐             ┌──────────────────┐
       │   Ollama +      │             │  Python Quant     │
       │   Llama 3.2 3B  │             │     Engine        │
       └─────────────────┘             └─────────┬────────┘
                │                                 │
                │                                 │
                ▼                                 ▼
       AI Reasoning Layer                 Deterministic Layer
       - Researcher                      - Indicators
       - Critic                          - Strategies
       - Validator                       - Backtesting
                                          - Metrics
                                          - Validation
                                          - Reporting

 Technology Stack:
Programming
- Python 3.11
Data & Quantitative Analysis
- pandas
- NumPy
- SciPy
- yfinance
- Backtesting.py
Backend
- FastAPI
- Uvicorn
- Pydantic
AI
- Ollama
- Llama 3.2 3B
- LangChain
Workflow Orchestration
- n8n
Reporting
- ReportLab
- Matplotlib
Communication
- Requests




Requirements
Recommended environment:
Python 3.11+
Ollama
n8n
Git

Clone the repository:
git clone https://github.com/Ayyjinx/Quantforge.git
cd Quantforge

Create a virtual environment:
python -m venv venv


Activate it on Windows:
.\venv\Scripts\Activate.ps1

Install the required Python packages:
pip install pandas numpy scipy yfinance backtesting fastapi uvicorn reportlab matplotlib requests pydantic

Install Ollama and pull the model:
ollama pull llama3.2:3b
QuantForge uses the local Ollama endpoint:
http://127.0.0.1:11434

Quantitative Engine
From the project directory:
uvicorn quant_engine.api:app --host 127.0.0.1 --port 8000

Progress Server
python progress_server.py

The progress server runs on:
http://127.0.0.1:8002

Input Gateway / Dashboard
python input_gateway.py

The dashboard runs on:
http://127.0.0.1:8001

n8n
Start n8n:
n8n

The n8n interface is available at:
http://localhost:5678

Import the workflow from:
Quantforge.json
