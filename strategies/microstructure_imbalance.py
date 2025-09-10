class MicrostructureImbalance:
    def __init__(self, df, window=10):
        self.df = df
        self.window = window

    def run(self):
        df = self.df.copy()
        trades = []
        df["return"] = df["Close"].pct_change()

        for i in range(self.window, len(df)):
            sign_sum = df["return"].iloc[i-self.window:i].apply(lambda x: 1 if x > 0 else -1).sum()
            if sign_sum > self.window/2:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
            elif sign_sum < -self.window/2:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
        return trades
