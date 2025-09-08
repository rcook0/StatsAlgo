import pandas as pd
import numpy as np

def run_strategy(df):
    if len(df)<20:
        return None
    sample = df['close'].pct_change().dropna().sample(5, replace=True)
    if sample.mean()>0:
        return 'buy'
    elif sample.mean()<0:
        return 'sell'
    return None
