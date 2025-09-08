import pandas as pd

def run_strategy(df):
    if len(df)<4:
        return None
    returns = df['close'].pct_change().fillna(0)
    last_seq = returns[-3:]
    if all(last_seq>0):
        return 'sell'
    elif all(last_seq<0):
        return 'buy'
    return None
