import pandas as pd

def run_strategy(df):
    if len(df)<2:
        return None
    prev_close = df['close'].iloc[-2]
    curr_open = df['open'].iloc[-1]
    gap = curr_open-prev_close
    if gap>0.01*prev_close:
        return 'sell'
    elif gap<-0.01*prev_close:
        return 'buy'
    return None
