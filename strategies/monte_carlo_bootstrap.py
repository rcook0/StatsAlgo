import numpy as np

class MonteCarloBootstrap:
    def __init__(self, df, sample_size=50):
        self.df = df
        self.sample_size = sample_size

    def run(self):
        df = self.df.copy()
        trades = []
        returns = df["Close"].pct_change().dropna()
        sampled = np.random.choice(returns, size=min(self.sample_size, len(returns)), replace=True)
        avg_return = sampled.mean()

        for i in range(1, len(df)):
            if avg_return > 0:
                trades.append((df["Date"].iloc[i], "LONG", df["Close"].iloc[i]))
            else:
                trades.append((df["Date"].iloc[i], "SHORT", df["Close"].iloc[i]))
        return trades
