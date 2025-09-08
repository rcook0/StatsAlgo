# Opening Range Breakout
import pandas as pd

def run_strategy(df):
    if len(df)<30:
        return None
    open_range = df.iloc[:30]
    high = open_range['high'].max()
    low = open_range['low'].min()
    last = df.iloc[-1]
    if last['close']>high:
        return 'buy'
    elif last['close']<low:
        return 'sell'
    return None
