class ProbabilisticTimeEdge:
    def __init__(self, df, hour=10):
        self.df = df
        self.hour = hour

    def run(self):
        df = self.df.copy()
        trades = []
        subset = df[df["Date"].dt.hour == self.hour]
        direction = "LONG" if subset["Close"].mean() > subset["Open"].mean() else "SHORT"

        for _, row in subset.iterrows():
            trades.append((row["Date"], direction, row["Close"]))
        return trades
