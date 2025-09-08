# sigma_extreme.py
"""
Sigma Extreme Strategy

Signals when returns deviate more than N sigma from mean.
"""
import pandas as pd
import numpy as np

def run_strategy(df):
    if len(df)<20:
        return None
    df['returns'] = df['close'].pct_change()
    mean = df['returns'].mean()
    std = df['returns'].std()
    latest = df['returns'].iloc[-1]
    if latest > mean + 2*std:
        return 'sell'
    elif latest < mean - 2*std:
        return 'buy'
    return None
