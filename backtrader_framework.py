import backtrader as bt
import pandas as pd
from datetime import time as dtime
import plotly.graph_objects as go
import numpy as np

# =========================
# Base class
# =========================
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
        p, b = self.p.p_win, self.p.payoff_ratio
        q = 1 - p
        f = (b*p - q)/b
        f = max(0, min(f, 0.25))
        equity = self.broker.getvalue()
        return equity*f/self.data.close[0]

# =========================
# Strategy 1: Morning Z-Score
# =========================
class MorningZScore(BaseStatStrategy):
    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        t = self.datas[0].datetime.time(0)
        if t==dtime(0,0): self.executed_today=False
        if t==dtime(7,0) and not self.executed_today:
            self.executed_today=True
            if len(self.data) > self.p.lookback:
                closes=pd.Series(self.data.close.get(size=self.p.lookback))
                mean,std=closes.mean(),closes.std()
                z=(self.data.close[0]-mean)/std if std>0 else 0
                if z>self.p.sigma_threshold:
                    self.sell(size=self.kelly_size()); self.trades.append((dt,self.data.close[0],'SELL'))
                elif z<-self.p.sigma_threshold:
                    self.buy(size=self.kelly_size()); self.trades.append((dt,self.data.close[0],'BUY'))

# =========================
# Strategy 2: Overnight Gap Reversion
# =========================
class OvernightGapReversion(BaseStatStrategy):
    def next(self):
        dt=self.datas[0].datetime.datetime(0)
        t=self.datas[0].datetime.time(0)
        if t==dtime(0,0): self.executed_today=False
        if t==dtime(7,0) and not self.executed_today and len(self.data)>2:
            self.executed_today=True
            prev_close=self.data.close[-1]
            gap=self.data.open[0]-prev_close
            ret_series=pd.Series([self.data.close[i+1]-self.data.close[i] for i in range(-self.p.lookback,-1)])
            mean_gap,std_gap=ret_series.mean(),ret_series.std()
            if gap>mean_gap+self.p.sigma_threshold*std_gap:
                self.sell(size=self.kelly_size()); self.trades.append((dt,self.data.close[0],'SELL'))
            elif gap<mean_gap-self.p.sigma_threshold*std_gap:
                self.buy(size=self.kelly_size()); self.trades.append((dt,self.data.close[0],'BUY'))

# =========================
# Strategy 3: Sequential Reversal
# =========================
class SequentialReversal(BaseStatStrategy):
    def next(self):
        dt=self.datas[0].datetime.datetime(0)
        t=self.datas[0].datetime.time(0)
        if t==dtime(0,0): self.executed_today=False
        if t==dtime(7,0) and not self.executed_today and len(self.data)>3:
            self.executed_today=True
            # count last 3 bars direction
            up_down = [1 if self.data.close[-i]-self.data.close[-i-1]>0 else -1 for i in range(3)]
            if all(d>0 for d in up_down):  # 3 up bars
                self.sell(size=self.kelly_size()); self.trades.append((dt,self.data.close[0],'SELL'))
            elif all(d<0 for d in up_down):  # 3 down bars
                self.buy(size=self.kelly_size()); self.trades.append((dt,self.data.close[0],'BUY'))

# =========================
# Strategy 4: Monte Carlo Probabilistic Return
# =========================
class MonteCarloProb(BaseStatStrategy):
    def next(self):
        dt=self.datas[0].datetime.datetime(0)
        t=self.datas[0].datetime.time(0)
        if t==dtime(0,0): self.executed_today=False
        if t==dtime(7,0) and not self.executed_today and len(self.data)>self.p.lookback:
            self.executed_today=True
            returns=np.diff(self.data.close.get(size=self.p.lookback))/self.data.close.get(size=self.p.lookback)[:-1]
            sim=np.random.choice(returns, size=1000, replace=True)
            prob_up=(sim>0).mean()
            if prob_up>0.6: self.buy(size=self.kelly_size()); self.trades.append((dt,self.data.close[0],'BUY'))
            elif prob_up<0.4: self.sell(size=self.kelly_size()); self.trades.append((dt,self.data.close[0],'SELL'))

# =========================
# CSV Data Loader
# =========================
class GenericCSVData(bt.feeds.GenericCSVData):
    params=(('fromdate',None),('todate',None),('nullvalue',0.0),
            ('dtformat','%Y-%m-%d %H:%M:%S'),('datetime',0),('open',1),
            ('high',2),('low',3),('close',4),('volume',5),('openinterest',-1))

# =========================
# Walk-Forward Test
# =========================
def walk_forward_plotly(csv_file, strategy_class, train_days=30, test_days=5):
    df=pd.read_csv(csv_file,parse_dates=['datetime']); df.set_index('datetime',inplace=True)
    cumulative_equity=[]; all_trades=[]; equity_base=100000
    start_idx=0; total_bars=len(df); bar_per_day=24*60

    while start_idx + train_days*bar_per_day + test_days*bar_per_day <= total_bars:
        train_df=df.iloc[start_idx:start_idx+train_days*bar_per_day]
        test_df=df.iloc[start_idx+train_days*bar_per_day:start_idx+(train_days+test_days)*bar_per_day]

        # Optimize
        cerebro_train=bt.Cerebro(optreturn=False)
        train_data=GenericCSVData(dataname=train_df.reset_index())
        cerebro_train.adddata(train_data)
        cerebro_train.optstrategy(strategy_class,
                                  sigma_threshold=[0.8,1.0,1.2],
                                  lookback=[10,20,30])
        cerebro_train.broker.setcash(equity_base)
        train_results=cerebro_train.run(maxcpus=1)
        best_val,best_params=None,None
        for run in train_results:
            for strat in run:
                val=strat.log_value; params=(strat.p.sigma_threshold,strat.p.lookback)
                if best_val is None or val>best_val: best_val,val=params,strat.log_value

        # Test
        cerebro_test=bt.Cerebro(); test_data=GenericCSVData(dataname=test_df.reset_index())
        cerebro_test.adddata(test_data)
        cerebro_test.addstrategy(strategy_class,sigma_threshold=best_params[0],lookback=best_params[1])
        cerebro_test.broker.setcash(equity_base)
        test_result=cerebro_test.run(); final_val=test_result[0].log_value
        trades=test_result[0].trades
        all_trades.extend(trades); equity_base=final_val; cumulative_equity.append(final_val)
        start_idx+=test_days*bar_per_day

    # Plot interactive
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df['close'],mode='lines',name='Close Price'))
    for dt,price,side in all_trades:
        color='green' if side=='BUY' else 'red'; symbol='triangle-up' if side=='BUY' else 'triangle-down'
        fig.add_trace(go.Scatter(x=[dt],y=[price],mode='markers',
                                 marker=dict(symbol=symbol,color=color,size=10),
                                 name=f'{side}',text=f'{side} @ {price:.2f} on {dt}',showlegend=False))
    fig.add_trace(go.Scatter(x=[i for i in range(len(cumulative_equity))],y=cumulative_equity,
                             mode='lines+markers',name='Cumulative Equity',line=dict(color='blue',width=2)))
    fig.update_layout(title="Walk-Forward Statistical Strategy with Trades & Equity",
                      xaxis_title="Date / Walk-Forward Periods",yaxis_title="Price / Portfolio Value",
                      hovermode='closest')
    fig.show()

# =========================
# Example Usage
# =========================
if __name__=="__main__":
    walk_forward_plotly("US30_Backtrader_Template.csv",MorningZScore,train_days=30,test_days=5)
    walk_forward_plotly("US30_Backtrader_Template.csv",OvernightGapReversion,train_days=30,test_days=5)
    walk_forward_plotly("US30_Backtrader_Template.csv",SequentialReversal,train_days=30,test_days=5)
    walk_forward_plotly("US30_Backtrader_Template.csv",MonteCarloProb,train_days=30,test_days=5)
