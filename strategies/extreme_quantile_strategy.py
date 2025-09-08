import pandas as pd

def run_strategy(df):
    if len(df)<20:
        return None
    q_high = df['close'].quantile(0.95)
    q_low = df['close'].quantile(0.05)
    last = df['close'].iloc[-1]
    if last>q_high:
        return 'sell'
    elif last<q_low:
        return 'buy'
    return None
