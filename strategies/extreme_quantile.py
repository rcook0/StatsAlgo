import numpy as np

class ExtremeQuantile:
    def __init__(self, df, q=0.95):
        self.df = df
        self.q = q

    def run(self):
        df = self.df.copy()
        trades = []
        q_high = df["Close"].quantile(self.q)
        q_low = df["Close"].quantile(1 - self.q)

        for i in range(len(df)):
            if df["Close"].iloc[i] > q_high:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
            elif df["Close"].iloc[i] < q_low:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
        return trades
