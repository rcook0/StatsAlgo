"""
Demo OHLCV Data Generator
-------------------------
Generates a random-walk OHLCV dataset suitable for testing the backtester.
Default: 1000 rows of 1-minute bars.

Columns:
- Date (timestamp)
- Open
- High
- Low
- Close
- Volume

Usage:
    python generate_demo_data.py

Outputs:
    demo_data.csv (ready for app.py upload)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_demo_data(rows=1000, start_price=35000, seed=42):
    np.random.seed(seed)
    
    dates = [datetime(2023, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(rows)]
    prices = [start_price]

    # Random walk for close prices
    for _ in range(1, rows):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.0005)))  # ~0.05% std

    # Create OHLC from close
    df = pd.DataFrame({"Date": dates})
    df["Close"] = prices
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + np.random.uniform(0, 0.001, rows))
    df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - np.random.uniform(0, 0.001, rows))
    df["Volume"] = np.random.randint(100, 5000, size=rows)

    return df

if __name__ == "__main__":
    df = generate_demo_data()
    df.to_csv("demo_data.csv", index=False)
    print("✅ Demo data generated: demo_data.csv")
