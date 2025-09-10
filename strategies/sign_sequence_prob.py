class SignSequenceProb:
    def __init__(self, df, seq_len=3):
        self.df = df
        self.seq_len = seq_len

    def run(self):
        df = self.df.copy()
        trades = []
        df["return"] = df["Close"].pct_change()

        for i in range(self.seq_len, len(df)):
            seq = (df["return"].iloc[i-self.seq_len:i] > 0).astype(int).tolist()
            if sum(seq) == self.seq_len:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
            elif sum(seq) == 0:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
        return trades
