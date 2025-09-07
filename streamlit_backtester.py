"""
streamlit_backtester.py

Interactive backtesting UI (Streamlit + Plotly)
- Scans ./output/ for symbol folders (output/<SYMBOL>/<TIMEFRAME>.csv)
- Choose symbol/timeframe, date/time start, strategy, sizing
- Step bar-by-bar or Play at a selected speed
- Shows price chart with buy/sell markers and live equity curve/trade log

Notes:
- This is a demonstration / research tool. It runs pure historical backtests
  in-memory and is not wired to any live brokerage.
- Strategies are implemented as simple step-functions; add / tweak as needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data
def list_symbols_and_timeframes(output_root="./output"):
    root = Path(output_root)
    symbols = []
    symbol_map = {}
    if not root.exists():
        return symbols, symbol_map
    for symbol_dir in root.iterdir():
        if symbol_dir.is_dir():
            tfs = []
            for f in symbol_dir.glob("*.csv"):
                # tf is filename without extension (e.g., "1min" or "1T")
                tf = f.stem
                tfs.append(tf)
            if tfs:
                symbols.append(symbol_dir.name)
                symbol_map[symbol_dir.name] = sorted(tfs)
    return symbols, symbol_map

@st.cache_data
def load_symbol_timeframe(symbol, timeframe, output_root="./output"):
    fn = Path(output_root) / symbol / f"{timeframe}.csv"
    if not fn.exists():
        # try alternative filename patterns
        candidates = list((Path(output_root)/symbol).glob(f"*{timeframe}*.csv"))
        if candidates:
            fn = candidates[0]
        else:
            raise FileNotFoundError(f"No file for {symbol}/{timeframe}")
    df = pd.read_csv(fn, parse_dates=True)
    # try to ensure a datetime index column exists
    # common headers: Date, DateTime, index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    elif 'Datetime' in df.columns or 'datetime' in df.columns:
        c = 'Datetime' if 'Datetime' in df.columns else 'datetime'
        df[c] = pd.to_datetime(df[c])
        df = df.set_index(c)
    else:
        # assume first column is datetime-like
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
        df = df.set_index(df.columns[0])
    # Ensure standard OHLCV columns exist (capitalize)
    df.columns = [c.capitalize() for c in df.columns]
    for col in ['Open','High','Low','Close','Volume']:
        if col not in df.columns:
            df[col] = 0.0
    df = df[['Open','High','Low','Close','Volume']].copy()
    df.sort_index(inplace=True)
    return df

# ---------------------------
# Kelly & sizing helper
# ---------------------------
def kelly_fraction(p, b):
    """
    Kelly fraction f* = (b*p - q)/b  (b = payoff ratio, q = 1-p)
    Caps between 0 and 0.25 by design here (user can change)
    """
    q = 1 - p
    f = (b * p - q) / b if b != 0 else 0.0
    return max(0.0, min(f, 0.25))

# ---------------------------
# Strategy implementations (simple, stat/prob-based)
# Each strategy is a function: (df_history) -> signal for the last bar {1, -1, 0}
# Keep logic lightweight and readable; add more strategies as required.
# ---------------------------
def strat_morning_zscore(df_hist, lookback=20, sigma=1.0):
    # uses last lookback closes, signal based on zscore of last close
    if len(df_hist) < lookback:
        return 0
    closes = df_hist['Close'].iloc[-lookback:]
    mean = closes.mean(); std = closes.std()
    if std == 0 or pd.isna(std):
        return 0
    z = (closes.iloc[-1] - mean) / std
    if z > sigma:
        return -1
    elif z < -sigma:
        return 1
    return 0

def strat_overnight_gap(df_hist, lookback=20, sigma=1.0):
    # gap = open - previous close; use recent gaps' stats
    if len(df_hist) < lookback + 1:
        return 0
    prev_close = df_hist['Close'].iloc[-2]
    gap = df_hist['Open'].iloc[-1] - prev_close
    # historical gaps
    prev_gaps = df_hist['Open'].iloc[-(lookback+1):-1].values - df_hist['Close'].iloc[-(lookback+1):-1].values
    mean, std = np.mean(prev_gaps), np.std(prev_gaps)
    if std == 0: return 0
    if gap > mean + sigma * std:
        return -1
    elif gap < mean - sigma * std:
        return 1
    return 0

def strat_sequential_reversal(df_hist, seq_len=3):
    # if last seq_len bars all up -> sell; all down -> buy
    if len(df_hist) < seq_len + 1:
        return 0
    diffs = np.sign(df_hist['Close'].diff().iloc[-seq_len:]).fillna(0)
    if (diffs > 0).all():
        return -1
    if (diffs < 0).all():
        return 1
    return 0

def strat_montecarlo_bootstrap(df_hist, lookback=50, sims=500, threshold_up=0.6):
    # bootstrap returns and check P(up)
    if len(df_hist) < lookback + 1:
        return 0
    closes = df_hist['Close'].iloc[-(lookback+1):].values
    rets = np.diff(closes) / closes[:-1]
    sim = np.random.choice(rets, size=sims, replace=True)
    p_up = (sim > 0).mean()
    if p_up > threshold_up:
        return 1
    if p_up < (1 - threshold_up):
        return -1
    return 0

def strat_vol_breakout(df_hist, lookback=20, sigma=1.2):
    # if current return is > sigma * std(prev returns) -> follow
    if len(df_hist) < lookback + 1:
        return 0
    closes = df_hist['Close'].iloc[-(lookback+1):].values
    prev_rets = np.diff(closes[:-1]) / closes[:-2]
    current_ret = (closes[-1] - closes[-2]) / closes[-2]
    std = np.std(prev_rets)
    if std == 0: return 0
    if current_ret > sigma * std:
        return 1
    if current_ret < -sigma * std:
        return -1
    return 0

# map strategy names
STRATEGIES = {
    "Morning Z-Score": strat_morning_zscore,
    "Overnight Gap Reversion": strat_overnight_gap,
    "Sequential Reversal": strat_sequential_reversal,
    "Monte Carlo Bootstrap": strat_montecarlo_bootstrap,
    "Volatility Breakout": strat_vol_breakout
}

# ---------------------------
# Portfolio engine (simple, mark-to-market)
# ---------------------------
class SimplePortfolio:
    def __init__(self, start_cash=100000.0, kelly=False, p_win=0.55, payoff=1.2, frac=0.01, kelly_cap=0.25):
        self.start_cash = float(start_cash)
        self.cash = float(start_cash)
        self.position = 0.0        # units (positive for long, negative for short)
        self.entry_price = None
        self.equity_history = []
        self.trades = []           # list of dicts
        self.kelly = kelly
        self.p_win = p_win
        self.payoff = payoff
        self.frac = frac          # fixed fractional sizing if not Kelly (fraction of equity)
        self.kelly_cap = kelly_cap

    def current_equity(self, price):
        return self.cash + self.position * price

    def compute_size(self, price):
        equity = self.current_equity(price)
        if self.kelly:
            f = kelly_fraction(self.p_win, self.payoff)
            f = min(f, self.kelly_cap)
        else:
            f = self.frac
        if f <= 0:
            return 0.0
        # units to buy: (equity * f) / price
        units = (equity * f) / price
        return units

    def enter_long(self, price, reason="", time=None):
        if self.position > 0:
            return  # already long
        units = self.compute_size(price)
        if units <= 0:
            return
        cost = units * price
        # if currently short, we first close short
        if self.position < 0:
            # cover short
            self.cash -= units * price  # buy back
        # buy
        self.position += units
        self.cash -= cost
        self.entry_price = price
        self.trades.append({"time": time, "side": "BUY", "price": price, "size": units, "reason": reason})

    def enter_short(self, price, reason="", time=None):
        if self.position < 0:
            return
        units = self.compute_size(price)
        if units <= 0:
            return
        # proceeds from short sale
        proceeds = units * price
        self.position -= units  # negative position
        self.cash += proceeds
        self.entry_price = price
        self.trades.append({"time": time, "side": "SELL_SHORT", "price": price, "size": units, "reason": reason})

    def exit_position(self, price, reason="", time=None):
        if self.position == 0:
            return
        if self.position > 0:
            # sell longs
            proceeds = self.position * price
            self.cash += proceeds
            self.trades.append({"time": time, "side": "SELL", "price": price, "size": self.position, "reason": reason})
            self.position = 0.0
        else:
            # cover shorts
            units = -self.position
            cost = units * price
            self.cash -= cost
            self.trades.append({"time": time, "side": "BUY_TO_COVER", "price": price, "size": units, "reason": reason})
            self.position = 0.0
        self.entry_price = None

    def record_equity(self, price):
        self.equity_history.append(self.current_equity(price))

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(layout="wide", page_title="Interactive Backtester")

st.title("Interactive Backtester — bar-by-bar stepping & play")

# Left panel: controls
with st.sidebar:
    st.header("Dataset & Strategy")
    symbols, symbol_map = list_symbols_and_timeframes()
    if not symbols:
        st.warning("No symbols found in ./output. Run the converter first and place results in ./output/<SYMBOL>/<TIMEFRAME>.csv")
        st.stop()

    symbol = st.selectbox("Symbol", symbols)
    tfs = symbol_map.get(symbol, [])
    timeframe = st.selectbox("Timeframe / File", tfs)

    df = load_symbol_timeframe(symbol, timeframe)

    # date/time range
    idx = df.index
    start_dt = st.date_input("Start date", value=idx[0].date(), min_value=idx[0].date(), max_value=idx[-1].date())
    # optional time selection
    start_time = st.time_input("Start time (optional)", value=idx[0].time())
    start_datetime = datetime.combine(start_dt, start_time)
    # find nearest index >= start_datetime
    if start_datetime.tzinfo is None:
        # naive treat as naive
        # find first index >=
        try:
            start_pos = df.index.get_loc(pd.to_datetime(start_datetime), method='nearest')
        except Exception:
            start_pos = 0
    else:
        start_pos = df.index.get_loc(pd.to_datetime(start_datetime), method='nearest')

    st.markdown("---")
    strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
    st.sidebar.markdown("Strategy parameters")
    lookback = st.sidebar.number_input("Lookback", value=20, min_value=2, max_value=1000)
    sigma = st.sidebar.number_input("Sigma / Threshold", value=1.0, format="%.2f")
    seq_len = st.sidebar.number_input("Sequence length", value=3, min_value=2)
    sims = st.sidebar.number_input("MC sims", value=500, min_value=50)
    st.sidebar.markdown("---")
    st.header("Position sizing")
    start_cash = st.number_input("Starting capital", value=100000.0, step=1000.0)
    use_kelly = st.checkbox("Use Kelly sizing", value=False)
    p_win = st.number_input("Kelly p_win", value=0.55, format="%.2f")
    payoff = st.number_input("Kelly payoff ratio b", value=1.2, format="%.2f")
    fixed_frac = st.number_input("Fixed fraction (if not Kelly)", value=0.02, format="%.4f", step=0.01)

    st.markdown("---")
    st.header("Playback controls")
    speed = st.slider("Bars per second (play)", 0.2, 10.0, 1.0)
    play = st.button("Play")
    pause = st.button("Pause")
    step_forward = st.button("Step ▶")
    step_back = st.button("◀ Step back")
    reset = st.button("Reset")

# initialize session state
if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = int(start_pos)
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = SimplePortfolio(start_cash, kelly=use_kelly, p_win=p_win, payoff=payoff, frac=fixed_frac)
if 'playing' not in st.session_state:
    st.session_state['playing'] = False
if 'markers' not in st.session_state:
    st.session_state['markers'] = []  # list of trades shown on plot

# handle reset
if reset:
    st.session_state['current_idx'] = int(start_pos)
    st.session_state['portfolio'] = SimplePortfolio(start_cash, kelly=use_kelly, p_win=p_win, payoff=payoff, frac=fixed_frac)
    st.session_state['markers'] = []

# handle step back
if step_back:
    if st.session_state['current_idx'] > 0:
        st.session_state['current_idx'] -= 1

# handle pause
if pause:
    st.session_state['playing'] = False

# handle play
if play:
    st.session_state['playing'] = True

# handle step forward (single bar)
if step_forward:
    st.session_state['current_idx'] += 1
    st.session_state['playing'] = False

# auto-play loop (careful with Streamlit rerun model)
if st.session_state['playing']:
    # Iterate until end or paused
    # We'll increment index a bit then rerun so UI remains responsive
    st.session_state['current_idx'] = min(st.session_state['current_idx'] + 1, len(df)-1)
    time.sleep(max(0.01, 1.0 / float(speed)))
    # trigger rerun
    st.experimental_rerun()

# Bound current index
st.session_state['current_idx'] = max(0, min(st.session_state['current_idx'], len(df)-1))
current_idx = st.session_state['current_idx']

# Strategy selection and parameter passing
strategy_func = STRATEGIES[strategy_name]

# Prepare the slice of data up to current index for display and strategy
df_upto = df.iloc[:current_idx+1].copy()

# compute signal for current bar
sig = 0
# choose parameters per strategy
if strategy_name == "Morning Z-Score":
    sig = strategy_func(df_upto, lookback=lookback, sigma=sigma)
elif strategy_name == "Overnight Gap Reversion":
    sig = strategy_func(df_upto, lookback=lookback, sigma=sigma)
elif strategy_name == "Sequential Reversal":
    sig = strategy_func(df_upto, seq_len=seq_len)
elif strategy_name == "Monte Carlo Bootstrap":
    sig = strategy_func(df_upto, lookback=lookback, sims=sims)
elif strategy_name == "Volatility Breakout":
    sig = strategy_func(df_upto, lookback=lookback, sigma=sigma)
else:
    sig = strategy_func(df_upto)

# Apply trading decision: for demo we treat signal as an instruction at this bar
bar_time = df_upto.index[-1]
bar_price = df_upto['Close'].iloc[-1]

# Simple trade logic: signal 1 -> go long (enter/flip); -1 -> go short (enter/flip); 0 -> hold
reason = f"{strategy_name}"
if sig == 1:
    # if opposite position, exit first
    if st.session_state['portfolio'].position < 0:
        st.session_state['portfolio'].exit_position(bar_price, reason="flip_to_long", time=bar_time)
        st.session_state['markers'].append((bar_time, bar_price, "COVER"))
    st.session_state['portfolio'].enter_long(bar_price, reason=reason, time=bar_time)
    st.session_state['markers'].append((bar_time, bar_price, "BUY"))
elif sig == -1:
    if st.session_state['portfolio'].position > 0:
        st.session_state['portfolio'].exit_position(bar_price, reason="flip_to_short", time=bar_time)
        st.session_state['markers'].append((bar_time, bar_price, "SELL"))
    st.session_state['portfolio'].enter_short(bar_price, reason=reason, time=bar_time)
    st.session_state['markers'].append((bar_time, bar_price, "SHORT"))

# record equity
st.session_state['portfolio'].record_equity(bar_price)

# ---------------------------
# Main display: Charts & logs
# ---------------------------
# Price chart with markers
price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=df_upto.index, y=df_upto['Close'], mode='lines', name='Close'))

# add markers from session stored trades
for t, p, tag in st.session_state['markers']:
    if t <= df_upto.index[-1]:
        color = 'green' if tag in ("BUY","COVER") else 'red'
        symbol = 'triangle-up' if tag in ("BUY","COVER") else 'triangle-down'
        price_fig.add_trace(go.Scatter(x=[t], y=[p], mode='markers', marker_symbol=symbol,
                                       marker=dict(size=12, color=color), name=tag))

price_fig.update_layout(title=f"{symbol} {timeframe} — up to {df_upto.index[-1]}", yaxis_title="Price")

# Equity chart
eq = st.session_state['portfolio'].equity_history
eq_x = df_upto.index[-len(eq):] if len(eq) > 0 else []
eq_fig = go.Figure()
if len(eq) > 0:
    eq_fig.add_trace(go.Scatter(x=eq_x, y=eq, mode='lines+markers', name='Equity'))
eq_fig.update_layout(title="Equity Curve", yaxis_title="Equity")

# Display side-by-side
col1, col2 = st.columns([3,2])
with col1:
    st.plotly_chart(price_fig, use_container_width=True)
with col2:
    st.plotly_chart(eq_fig, use_container_width=True)
    st.markdown("### Portfolio")
    st.write({
        "Cash": f"{st.session_state['portfolio'].cash:,.2f}",
        "Position units": st.session_state['portfolio'].position,
        "Last price": f"{bar_price:,.2f}",
        "Current equity": f"{st.session_state['portfolio'].current_equity(bar_price):,.2f}"
    })

# trades log & controls
st.markdown("---")
st.subheader("Trade log (most recent first)")
trades_df = pd.DataFrame(st.session_state['portfolio'].trades[::-1])
if not trades_df.empty:
    st.dataframe(trades_df)
else:
    st.write("No trades yet.")

# Full-run button: run strategy over whole selected date range fast and show summary
if st.button("Run full historical simulation (same strategy)"):
    # create fresh portfolio for full run
    pf = SimplePortfolio(start_cash, kelly=use_kelly, p_win=p_win, payoff=payoff, frac=fixed_frac)
    markers = []
    full_df = df.copy()
    for i in range(len(full_df)):
        hist = full_df.iloc[:i+1]
        # compute signal using same selected params
        if strategy_name == "Morning Z-Score":
            s = strat_morning_zscore(hist, lookback=lookback, sigma=sigma)
        elif strategy_name == "Overnight Gap Reversion":
            s = strat_overnight_gap(hist, lookback=lookback, sigma=sigma)
        elif strategy_name == "Sequential Reversal":
            s = strat_sequential_reversal(hist, seq_len=seq_len)
        elif strategy_name == "Monte Carlo Bootstrap":
            s = strat_montecarlo_bootstrap(hist, lookback=lookback, sims=sims)
        elif strategy_name == "Volatility Breakout":
            s = strat_vol_breakout(hist, lookback=lookback, sigma=sigma)
        else:
            s = 0
        price = hist['Close'].iloc[-1]
        ttime = hist.index[-1]
        if s == 1:
            if pf.position < 0:
                pf.exit_position(price, reason="flip_to_long", time=ttime)
            pf.enter_long(price, reason=strategy_name, time=ttime)
        elif s == -1:
            if pf.position > 0:
                pf.exit_position(price, reason="flip_to_short", time=ttime)
            pf.enter_short(price, reason=strategy_name, time=ttime)
        pf.record_equity(price)
    # summary
    final_eq = pf.equity_history[-1] if pf.equity_history else pf.current_equity(full_df['Close'].iloc[-1])
    st.success(f"Full-run complete. Final equity: {final_eq:,.2f}")
    # show equity and trade count
    st.write({
        "Starting cash": start_cash,
        "Final equity": final_eq,
        "Number of trades": len(pf.trades)
    })
    # plot equity
    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(x=full_df.index[-len(pf.equity_history):], y=pf.equity_history, mode='lines', name='Equity'))
    st.plotly_chart(fig_full, use_container_width=True)

st.markdown("### Notes & next steps")
st.markdown("""
- Strategies here are simple illustrative versions of the statistical strategies we discussed.
- You can add more advanced entry/exit rules, stop-loss, take-profit, and trailing logic.
- For high-frequency datasets or large horizons, use the "Run full historical simulation" option for faster batch runs instead of stepping.
- This UI is meant for local research. If you want a web-hosted UI or integration with Backtrader engine, I can help adapt it.
""")
