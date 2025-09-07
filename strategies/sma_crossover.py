# This is a simple SMA crossover strategy
# Buy when SMA10 crosses above SMA50, Sell when SMA10 crosses below SMA50

import pandas as pd

def generate_signal(df: pd.DataFrame):
    """
    df: Historical data up to current bar
    Must return 1 (Buy), -1 (Sell), 0 (Hold)
    """
    if len(df) < 50:  # Need at least 50 bars
        return 0

    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    if df['SMA10'].iloc[-2] < df['SMA50'].iloc[-2] and df['SMA10'].iloc[-1] > df['SMA50'].iloc[-1]:
        return 1  # Buy signal
    elif df['SMA10'].iloc[-2] > df['SMA50'].iloc[-2] and df['SMA10'].iloc[-1] < df['SMA50'].iloc[-1]:
        return -1  # Sell signal
    else:
        return 0
