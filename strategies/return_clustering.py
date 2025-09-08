import pandas as pd
import numpy as np

def run_strategy(df):
    if len(df)<10:
        return None
    returns = df['close'].pct_change().fillna(0)
    recent_std = returns[-5:].std()
    overall_std = returns.std()
    if recent_std>1.5*overall_std:
        return 'buy'
    elif recent_std<0.5*overall_std:
        return 'sell'
    return None
