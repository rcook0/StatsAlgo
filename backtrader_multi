import backtrader as bt
import pandas as pd
from datetime import time as dtime
import matplotlib.pyplot as plt

# =======================
# Base Class for Stats Strategies
# =======================
class BaseStatStrategy(bt.Strategy):
    params = dict(
        lookback=20,
        sigma_threshold=1.0,
        p_win=0.55,
        payoff_ratio=1.2
    )

    def __init__(self):
        self.executed_today = False
        self.trades = []

    def kelly_size(self):
        p = self.p.p_win
        b = self.p.payoff_ratio
        q = 1 - p
        f = (b*p - q) / b
        f = max(0, min(f, 0.25))  # cap Kelly
        equity = self.broker.getvalue()
        risk_amount = equity * f
        return risk_amount / self.data.close[0]

# =======================
# Strategy 1: Morning 07:00 Z-score
# =======================
class MorningZScore(BaseStatStrategy):
    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        current_time = self.datas[0].datetime.time(0)
        if current_time == dtime(0,0):
            self.executed_today = False

        if current_time == dtime(7,0) and not self.executed_today:
            self.executed_today = True
            if len(self.data) > self.p.lookback:
                closes = pd.Series(self.data.close.get(size=self.p.lookback))
                mean = closes.mean()
                std = closes.std()
                deviation = (self.data.close[0] - mean) / std if std>0 else 0
                if deviation > self.p.sigma_threshold:
                    self.sell(size=self.kelly_size())
                    self.trades.append((dt, self.data.close[0], 'SELL'))
                elif deviation < -self.p.sigma_threshold:
                    self.buy(size=self.kelly_size())
                    self.trades.append((dt, self.data.close[0], 'BUY'))

# =======================
# Strategy 2: Overnight Gap Reversion
# =======================
class OvernightGapReversion(BaseStatStrategy):
    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        current_time = self.datas[0].datetime.time(0)
        if current_time == dtime(0,0):
            self.executed_today = False

        if current_time == dtime(7,0) and not self.executed_today:
            self.executed_today = True
            if len(self.data) > 2:
                prev_close = self.data.close[-1]
                gap = self.data.open[0] - prev_close
                mean_gap = pd.Series([self.data.close[i+1]-self.data.close[i] for i in range(-self.p.lookback, -1)]).mean()
                std_gap = pd.Series([self.data.close[i+1]-self.data.close[i] for i in range(-self.p.lookback, -1)]).std()
                if gap > mean_gap + self.p.sigma_threshold*std_gap:
                    self.sell(size=self.kelly_size())
                    self.trades.append((dt, self.data.close[0], 'SELL'))
                elif gap < mean_gap - self.p.sigma_threshold*std_gap:
                    self.buy(size=self.kelly_size())
                    self.trades.append((dt, self.data.close[0], 'BUY'))

# =======================
# CSV Data Loader
# =======================
class GenericCSVData(bt.feeds.GenericCSVData):
    params = (
        ('fromdate', None),
        ('todate', None),
        ('nullvalue', 0.0),
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
    )

# =======================
# Walk-forward Framework
# =======================
def walk_forward_test(csv_file, strategy_class, train_days=30, test_days=5):
    df = pd.read_csv(csv_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    cumulative_equity = []
    all_trades = []
    equity_base = 100000

    start_idx = 0
    total_bars = len(df)
    bar_per_day = 24*60  # adjust if not 1-min bars

    while start_idx + train_days*bar_per_day + test_days*bar_per_day <= total_bars:
        train_df = df.iloc[start_idx : start_idx + train_days*bar_per_day]
        test_df = df.iloc[start_idx + train_days*bar_per_day :
                          start_idx + (train_days + test_days)*bar_per_day]

        # --- Optimization on training ---
        cerebro_train = bt.Cerebro(optreturn=False)
        train_data = GenericCSVData(dataname=train_df.reset_index())
        cerebro_train.adddata(train_data)
        cerebro_train.optstrategy(
            strategy_class,
            sigma_threshold=[0.8,1.0,1.2,1.5,2.0],
            lookback=[10,20,30,40]
        )
        cerebro_train.broker.setcash(equity_base)
        train_results = cerebro_train.run(maxcpus=1)

        best_val, best_params = None, None
        for run in train_results:
            for strat in run:
                val = strat.log_value
                params = (strat.p.sigma_threshold, strat.p.lookback)
                if best_val is None or val > best_val:
                    best_val = val
                    best_params = params

        # --- Test ---
        cerebro_test = bt.Cerebro()
        test_data = GenericCSVData(dataname=test_df.reset_index())
        cerebro_test.adddata(test_data)
        cerebro_test.addstrategy(strategy_class,
                                 sigma_threshold=best_params[0],
                                 lookback=best_params[1])
        cerebro_test.broker.setcash(equity_base)
        test_result = cerebro_test.run()
        final_val = test_result[0].log_value
        trades = test_result[0].trades
        all_trades.extend(trades)

        equity_base = final_val
        cumulative_equity.append(final_val)
        print(f"Train σ={best_params[0]} lookback={best_params[1]} → Test final={final_val:.2f}")

        start_idx += test_days*bar_per_day

    return cumulative_equity, all_trades, df

# =======================
# Plot Function
# =======================
def plot_equity_and_trades(cumulative_equity, all_trades, df):
    plt.figure(figsize=(14,6))
    plt.subplot(2,1,1)
    plt.plot(range(len(cumulative_equity)), cumulative_equity, marker='o', color='blue')
    plt.title("Walk-Forward Cumulative Equity")
    plt.ylabel("Portfolio Value")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(df.index, df['close'], color='black', label='Close Price')
    for dt, price, side in all_trades:
        if side=='BUY':
            plt.scatter(dt, price, marker='^', color='green', label='BUY')
        else:
            plt.scatter(dt, price, marker='v', color='red', label='SELL')
    plt.title("Price with Trade Signals")
    plt.ylabel("Price")
    plt.xlabel("Datetime")
    plt.grid(True)
    plt.show()

# =======================
# Example Run
# =======================
if __name__ == "__main__":
    cum_eq, trades, df_data = walk_forward_test(
        "US30_Backtrader_Template.csv",
        strategy_class=MorningZScore,
        train_days=30,
        test_days=5
    )
    plot_equity_and_trades(cum_eq, trades, df_data)
