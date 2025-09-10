class TimeOfDayMeanReversion:
    def __init__(self, df, hour=14):
        self.df = df
        self.hour = hour

    def run(self):
        df = self.df.copy()
        trades = []
        hourly = df[df["Date"].dt.hour == self.hour]

        for i in range(1, len(hourly)):
            if hourly["Close"].iloc[i] > hourly["Close"].iloc[i-1]:
                trades.append((hourly["Date"].iloc[i], "SHORT", hourly["Close"].iloc[i]))
            else:
                trades.append((hourly["Date"].iloc[i], "LONG", hourly["Close"].iloc[i]))
        return trades
