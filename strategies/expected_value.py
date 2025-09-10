class ExpectedValue:
    def __init__(self, df, threshold=0.0005):
        self.df = df
        self.threshold = threshold

    def run(self):
        df = self.df.copy()
        trades = []
        df["return"] = df["Close"].pct_change()

        for i in range(1, len(df)):
            if df["return"].iloc[i] > self.threshold:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
            elif df["return"].iloc[i] < -self.threshold:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
        return trades
