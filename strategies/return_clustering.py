class ReturnClustering:
    def __init__(self, df, window=10):
        self.df = df
        self.window = window

    def run(self):
        df = self.df.copy()
        trades = []
        df["vol"] = df["Close"].pct_change().rolling(self.window).std()

        for i in range(self.window, len(df)):
            if df["vol"].iloc[i] > df["vol"].iloc[i-1]:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
            elif df["vol"].iloc[i] < df["vol"].iloc[i-1]:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
        return trades
