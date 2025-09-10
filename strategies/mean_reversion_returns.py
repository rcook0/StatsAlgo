class MeanReversionReturns:
    def __init__(self, df, lookback=5):
        self.df = df
        self.lookback = lookback

    def run(self):
        df = self.df.copy()
        trades = []
        df["return"] = df["Close"].pct_change()

        for i in range(self.lookback, len(df)):
            avg_return = df["return"].iloc[i-self.lookback:i].mean()
            if avg_return > 0 and df["return"].iloc[i] < 0:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
            elif avg_return < 0 and df["return"].iloc[i] > 0:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
        return trades
