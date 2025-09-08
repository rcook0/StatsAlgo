import pandas as pd

def run_strategy(df):
    if len(df)<60:
        return None
    hour = df['datetime'].iloc[-1].hour
    if hour in range(14,16):  # afternoon edge
        return 'buy'
    elif hour in range(10,12):  # morning edge
        return 'sell'
    return None
