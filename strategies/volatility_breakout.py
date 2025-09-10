import pandas as pd

class VolatilityBreakout:
    def __init__(self, df, k=0.5, lookback=20):
        self.df = df
        self.k = k
        self.lookback = lookback

    def run(self):
        df = self.df.copy()
        df["range"] = df["High"] - df["Low"]
        df["atr"] = df["range"].rolling(self.lookback).mean()
        trades = []

        for i in range(self.lookback, len(df)):
            breakout_up = df["Open"].iloc[i] + self.k * df["atr"].iloc[i]
            breakout_down = df["Open"].iloc[i] - self.k * df["atr"].iloc[i]
            if df["Close"].iloc[i] > breakout_up:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
            elif df["Close"].iloc[i] < breakout_down:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
        return trades
