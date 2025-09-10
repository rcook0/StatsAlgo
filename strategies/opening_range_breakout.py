import pandas as pd

class OpeningRangeBreakout:
    def __init__(self, df, minutes=30):
        self.df = df
        self.minutes = minutes

    def run(self):
        df = self.df.copy()
        trades = []

        day_groups = df.groupby(df["Date"].dt.date)
        for _, group in day_groups:
            open_period = group.iloc[:self.minutes]
            high, low = open_period["High"].max(), open_period["Low"].min()
            for i in range(self.minutes, len(group)):
                row = group.iloc[i]
                if row["Close"] > high:
                    trades.append((row["Date"], "LONG", row["Close"]))
                    break
                elif row["Close"] < low:
                    trades.append((row["Date"], "SHORT", row["Close"]))
                    break
        return trades
