import pandas as pd
import numpy as np

class SigmaExtreme:
    def __init__(self, df, window=20, sigma=2):
        self.df = df
        self.window = window
        self.sigma = sigma

    def run(self):
        df = self.df.copy()
        df["mean"] = df["Close"].rolling(self.window).mean()
        df["std"] = df["Close"].rolling(self.window).std()
        trades = []

        for i in range(self.window, len(df)):
            price, mean, std = df["Close"].iloc[i], df["mean"].iloc[i], df["std"].iloc[i]
            if price > mean + self.sigma * std:
                trades.append((df["Date"].iloc[i], "SHORT", price))
            elif price < mean - self.sigma * std:
                trades.append((df["Date"].iloc[i], "LONG", price))

        return trades
