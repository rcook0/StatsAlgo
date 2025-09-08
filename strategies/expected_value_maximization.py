import pandas as pd

def run_strategy(df):
    if len(df)<20:
        return None
    # simple EV: recent avg move
    ev = df['close'].diff().mean()
    if ev>0:
        return 'buy'
    elif ev<0:
        return 'sell'
    return None
