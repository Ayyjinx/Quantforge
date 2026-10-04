import yfinance as yf

print("Downloading EUR/USD data...")

data = yf.download(
    "EURUSD=X",
    period="2y",
    interval="1h",
    auto_adjust=False
)

# Remove the extra ticker level created by yfinance
if hasattr(data.columns, "levels"):
    data.columns = data.columns.get_level_values(0)

# Remove unnecessary column
if "Adj Close" in data.columns:
    data = data.drop(columns=["Adj Close"])

# Save clean data
data.to_csv("data/eurusd_h1.csv")

print("Data saved successfully.")
print("Columns:", list(data.columns))
print("Number of rows:", len(data))