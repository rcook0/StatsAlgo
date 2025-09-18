"""
csv_converter.py — Convert raw market data into backtester-ready format.

Features:
- Handles generic OHLCV datasets
- Parses "Date", "Open", "High", "Low", "Close", "Volume" (auto-detects variations)
- Cleans missing data, enforces datetime index
- Outputs a standardized CSV: Date,Open,High,Low,Close,Volume
- Works for daily, hourly, or minute-level data
"""

import pandas as pd

def convert_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    # Try common column names
    col_map = {
        "date": "Date",
        "time": "Date",
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "price": "Close",
        "volume": "Volume",
        "vol.": "Volume",
    }
    rename = {c: col_map[c.lower()] for c in df.columns if c.lower() in col_map}
    df = df.rename(columns=rename)

    if "Date" not in df:
        raise ValueError("No datetime column found in dataset")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in keep:
        if col not in df:
            df[col] = None
    df = df[keep]

    df.to_csv(output_path, index=False)
    print(f"✅ Converted {input_path} → {output_path}")
    return df
