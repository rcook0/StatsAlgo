class SequentialReversal:
    def __init__(self, df, run_len=3):
        self.df = df
        self.run_len = run_len

    def run(self):
        df = self.df.copy()
        trades = []
        df["return"] = df["Close"].pct_change()
        streak = 0

        for i in range(1, len(df)):
            if df["return"].iloc[i] > 0:
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1

            if streak >= self.run_len:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
            elif streak <= -self.run_len:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
        return trades
