class OvernightGap:
    def __init__(self, df, threshold=0.01):
        self.df = df
        self.threshold = threshold

    def run(self):
        df = self.df.copy()
        trades = []
        df["PrevClose"] = df["Close"].shift(1)

        for i in range(1, len(df)):
            gap = (df["Open"].iloc[i] - df["PrevClose"].iloc[i]) / df["PrevClose"].iloc[i]
            if gap > self.threshold:
                trades.append((df["Date"].iloc[i], "LONG", df["Open"].iloc[i]))
            elif gap < -self.threshold:
                trades.append((df["Date"].iloc[i], "SHORT", df["Open"].iloc[i]))
        return trades
