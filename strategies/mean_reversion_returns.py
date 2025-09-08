import pandas as pd

def run_strategy(df):
    if len(df)<20:
        return None
    returns = df['close'].pct_change()
    if returns.iloc[-1]>returns.mean()+returns.std():
        return 'sell'
    elif returns.iloc[-1]<returns.mean()-returns.std():
        return 'buy'
    return None
