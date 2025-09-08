import pandas as pd

def run_strategy(df):
    if len(df)<2:
        return None
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    gap = curr['open'] - prev['close']
    if gap>0.001*prev['close']:
        return 'sell'
    elif gap<-0.001*prev['close']:
        return 'buy'
    return None
