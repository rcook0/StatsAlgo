import pandas as pd

def run_strategy(df):
    if len(df)<5:
        return None
    returns = df['close'].pct_change().fillna(0)
    seq = returns[-3:]
    if all(seq>0):
        return 'sell'
    elif all(seq<0):
        return 'buy'
    return None
