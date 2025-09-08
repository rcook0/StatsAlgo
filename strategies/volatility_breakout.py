import pandas as pd

def run_strategy(df):
    if len(df)<20:
        return None
    range_pct = (df['high'] - df['low']).pct_change().iloc[-1]
    if range_pct>0.02:
        return 'buy'
    elif range_pct<-0.02:
        return 'sell'
    return None
