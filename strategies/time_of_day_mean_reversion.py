import pandas as pd

def run_strategy(df):
    if len(df)<60:
        return None
    last = df.iloc[-1]
    mean_price = df['close'].mean()
    if last['close']>mean_price*1.01:
        return 'sell'
    elif last['close']<mean_price*0.99:
        return 'buy'
    return None
