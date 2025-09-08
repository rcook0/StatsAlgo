import pandas as pd

def run_strategy(df):
    if len(df)<10:
        return None
    buys = df['close']>df['open']
    sells = df['close']<df['open']
    if buys.sum()>sells.sum()*1.5:
        return 'buy'
    elif sells.sum()>buys.sum()*1.5:
        return 'sell'
    return None
