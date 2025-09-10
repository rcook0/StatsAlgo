"""
app.py - Integrated Streamlit Backtester (imports strategies from ./strategies)

Features:
- Upload CSV or Load Demo Data (random walk) in-app
- Dynamic strategy loading from ./strategies
  - Supports either: run_strategy(df_slice) OR a class Strategy with methods update(bar) or on_bar(bar)
- Step-through / Auto-play (1..30 bars/sec) with Pause
- Simple trade engine with position sizing (fraction of equity), running/unrealized pnl and realized pnl
- Plotly candlestick chart with buy/sell markers and optional strategy overlays
- Batch mode run (run all bars automatically)
- Live-feed placeholder (append new bars and the app will pick them up on rerun)
Usage:
- Place strategy files in ./strategies
- Start: `streamlit run app.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib.util
import os
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Callable

# -----------------------
# Config
# -----------------------
DATA_DIR = "data"
STRATEGIES_DIR = "strategies"
INITIAL_EQUITY = 100000.0
MAX_SPEED = 30  # bars/sec

# -----------------------
# Helpers
# -----------------------
def generate_demo_data(rows=1000, start_price=35000, seed=42):
    np.random.seed(seed)
    start_dt = datetime(2023,1,3,13,30)  # NY time example
    dates = [start_dt + timedelta(minutes=i) for i in range(rows)]
    prices = [start_price]
    for _ in range(1, rows):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.0006)))
    df = pd.DataFrame({
        "datetime": dates,
        "open": pd.Series(prices).shift(1).fillna(prices[0]),
        "close": prices,
    })
    # add high/low
    df["high"] = df[["open","close"]].max(axis=1) * (1 + np.random.uniform(0,0.0007,size=rows))
    df["low"] = df[["open","close"]].min(axis=1) * (1 - np.random.uniform(0,0.0007,size=rows))
    df["volume"] = np.random.randint(100,5000,size=rows)
    # ensure types
    df = df[["datetime","open","high","low","close","volume"]]
    return df

def load_csv_to_df(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=[0])
    # fix column names: accept 'Date' or 'datetime' or 'DateTime'
    cols = [c.lower() for c in df.columns]
    if 'datetime' not in cols and 'date' in cols:
        df.rename(columns={df.columns[cols.index('date')]: 'datetime'}, inplace=True)
    if 'datetime' not in df.columns:
        # fallback: assume first column is datetime
        df.rename(columns={df.columns[0]:'datetime'}, inplace=True)
    # standardize column names
    df.columns = [c.lower() for c in df.columns]
    # expect open,high,low,close,volume
    for required in ['open','high','low','close']:
        if required not in df.columns:
            st.error(f"Uploaded CSV missing required column: {required}")
            return pd.DataFrame()
    df = df[['datetime','open','high','low','close'] + ([col for col in ['volume'] if col in df.columns])]
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_strategies_from_folder(folder):
    strategies = {}
    if not os.path.exists(folder):
        os.makedirs(folder)
    for file in os.listdir(folder):
        if file.endswith(".py") and not file.startswith("__"):
            name = file[:-3]
            path = os.path.join(folder, file)
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                strategies[name] = mod
            except Exception as e:
                st.warning(f"Failed to load strategy {name}: {e}")
    return strategies

def get_signal_from_module(mod, df_slice, bar=None):
    """
    Tries the following, in order:
      - mod.run_strategy(df_slice) -> return 'buy'/'sell'/None
      - If mod has class Strategy: instantiate if needed and call instance.update(bar) or instance.on_bar(bar) (returns dict or string)
    For class-based strategies we create & cache an instance in mod._instance
    """
    # function style
    try:
        if hasattr(mod, "run_strategy") and callable(mod.run_strategy):
            return mod.run_strategy(df_slice)
        if hasattr(mod, "generate_signals") and callable(mod.generate_signals):
            # generate_signals returns series of signals for df_slice, we take last
            s = mod.generate_signals(df_slice)
            if isinstance(s, (pd.Series, pd.DataFrame)):
                # series of 'LONG'/'SHORT' or 'buy'/'sell'
                val = s.iloc[-1] if len(s)>0 else None
                return val
    except Exception as e:
        st.warning(f"Strategy function error: {e}")

    # class style
    try:
        if hasattr(mod, "Strategy"):
            inst = getattr(mod, "_instance", None)
            if inst is None:
                # try to instantiate with data or without
                try:
                    inst = mod.Strategy(df_slice)
                except TypeError:
                    inst = mod.Strategy()
                setattr(mod, "_instance", inst)
            # now call update or on_bar
            if bar is not None:
                if hasattr(inst, "on_bar") and callable(inst.on_bar):
                    return inst.on_bar(bar)
                if hasattr(inst, "update") and callable(inst.update):
                    # some classes return None and set internal state; attempt to call update and then read last signal attr
                    try:
                        ret = inst.update(bar)
                        if ret is not None:
                            return ret
                    except Exception as e:
                        st.warning(f"Error calling update on Strategy class: {e}")
                    # fallback: if instance has last_signal attr
                    if hasattr(inst, "last_signal"):
                        return inst.last_signal
            else:
                # if no bar provided, try get_signals or generate_signals
                if hasattr(inst, "get_signals"):
                    s = inst.get_signals()
                    if isinstance(s, (pd.Series, list, np.ndarray)):
                        return s[-1]
    except Exception as e:
        st.warning(f"Strategy class error: {e}")
    return None

# -----------------------
# Session state init
# -----------------------
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()
if 'signals' not in st.session_state:
    st.session_state['signals'] = pd.DataFrame(columns=['datetime','signal','price'])
if 'current_index' not in st.session_state:
    st.session_state['current_index'] = 0
if 'playing' not in st.session_state:
    st.session_state['playing'] = False
if 'equity' not in st.session_state:
    st.session_state['equity'] = INITIAL_EQUITY
if 'cash' not in st.session_state:
    st.session_state['cash'] = INITIAL_EQUITY
if 'position' not in st.session_state:
    st.session_state['position'] = 0.0  # units
if 'trade_log' not in st.session_state:
    st.session_state['trade_log'] = []

# -----------------------
# UI - Data load
# -----------------------
st.title("Statistical Backtester — app.py (imports strategies)")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload historical CSV (datetime, open, high, low, close, [volume])", type=['csv'])
    demo_btn = st.button("Load Demo Data")
    if uploaded:
        df_loaded = load_csv_to_df(uploaded)
        if not df_loaded.empty:
            st.session_state['data'] = df_loaded
            st.session_state['current_index'] = 0
            st.session_state['signals'] = pd.DataFrame(columns=['datetime','signal','price'])
            st.session_state['cash'] = INITIAL_EQUITY
            st.session_state['position'] = 0.0
            st.session_state['trade_log'] = []
            st.success("CSV loaded into session")
    elif demo_btn:
        st.session_state['data'] = generate_demo_data(rows=1200)
        st.session_state['current_index'] = 0
        st.session_state['signals'] = pd.DataFrame(columns=['datetime','signal','price'])
        st.session_state['cash'] = INITIAL_EQUITY
        st.session_state['position'] = 0.0
        st.session_state['trade_log'] = []
        st.success("Demo data generated and loaded")

with col2:
    strategies = load_strategies_from_folder(STRATEGIES_DIR)
    strategy_names = list(strategies.keys())
    st.write("Strategies found in folder:")
    for s in strategy_names:
        st.write("• " + s)

# show data info
if st.session_state['data'].empty:
    st.info("No data loaded — upload a CSV or click 'Load Demo Data'.")
    st.stop()

# -----------------------
# Strategy selection / config
# -----------------------
st.sidebar.header("Strategy & Backtest Controls")
strategy_choice = st.sidebar.selectbox("Select strategy (from ./strategies)", strategy_names)
strategy_module = strategies[strategy_choice]

# simple strategy parameters
risk_pct = st.sidebar.number_input("Risk per trade (% equity)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)/100.0
fixed_size_units = st.sidebar.number_input("Optional fixed size units (0 = use risk%)", min_value=0.0, value=0.0)
sl_atr_mult = st.sidebar.number_input("Stop Loss (ATR multiples, 0 = disabled)", min_value=0.0, value=0.0)
tp_atr_mult = st.sidebar.number_input("Take Profit (ATR multiples, 0 = disabled)", min_value=0.0, value=0.0)

# playback config
mode = st.sidebar.radio("Mode", ["Step-through","Auto-run"])
speed = st.sidebar.slider("Speed (bars/sec) for Auto-run", 1, MAX_SPEED, 5)

# -----------------------
# Simple ATR helper (used if strategy wants it)
# -----------------------
def compute_atr(series_high, series_low, series_close, n=14):
    high_low = series_high - series_low
    high_close = (series_high - series_close.shift()).abs()
    low_close = (series_low - series_close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    return atr

# -----------------------
# Trading logic helpers
# -----------------------
def enter_position(price, size_units):
    # buy size_units units at price (long). We don't support shorting for simplicity here.
    st.session_state['position'] += size_units
    st.session_state['cash'] -= size_units * price
    st.session_state['trade_log'].append({
        'datetime': st.session_state['data']['datetime'].iloc[st.session_state['current_index']],
        'action':'BUY',
        'price':price,
        'size':size_units,
        'cash':st.session_state['cash']
    })

def exit_position(price, size_units=None):
    # close some or all of current long position
    if size_units is None:
        size_units = st.session_state['position']
    size_units = min(size_units, st.session_state['position'])
    st.session_state['position'] -= size_units
    st.session_state['cash'] += size_units * price
    st.session_state['trade_log'].append({
        'datetime': st.session_state['data']['datetime'].iloc[st.session_state['current_index']],
        'action':'SELL',
        'price':price,
        'size':size_units,
        'cash':st.session_state['cash']
    })

def current_equity(latest_price):
    return st.session_state['cash'] + st.session_state['position'] * latest_price

# -----------------------
# Visualization placeholders
# -----------------------
chart_placeholder = st.container()
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
eq_text = metrics_col1.metric("Equity", f"${st.session_state['cash']:.2f}")
pl_text = metrics_col2.metric("Unrealized P/L", "$0.00")
pct_text = metrics_col3.metric("% Gain/Loss", "0.00%")

# signal log table placeholder
log_placeholder = st.container()

# -----------------------
# Main step / run loop
# -----------------------
data = st.session_state['data']
n_bars = len(data)
i = st.session_state['current_index']

# Precompute ATR if needed for stop/tp or visualization
data['atr_14'] = compute_atr(data['high'], data['low'], data['close'], n=14)

# Ensure signals DF present
signals_df = st.session_state['signals']

# Utility to update display (chart + metrics)
def update_display(idx):
    # prepare df slice for plotting
    df_plot = data.iloc[:idx+1].copy()
    # plot
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['datetime'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='Price'
    )])
    # overlay bands if present in module (e.g., volatility breakout may provide 'UpperBand'/'LowerBand' series)
    try:
        if hasattr(strategy_module, 'get_bands'):
            bands = strategy_module.get_bands(df_plot)
            if 'upper' in bands:
                fig.add_trace(go.Scatter(x=df_plot['datetime'], y=bands['upper'], mode='lines', name='UpperBand', line=dict(dash='dash')))
            if 'lower' in bands:
                fig.add_trace(go.Scatter(x=df_plot['datetime'], y=bands['lower'], mode='lines', name='LowerBand', line=dict(dash='dash')))
    except Exception:
        pass

    # draw buy/sell markers from signals_df
    if not signals_df.empty:
        buys = signals_df[signals_df['signal'].isin(['buy','LONG','LONG_ENTRY','enter_long'])]
        sells = signals_df[signals_df['signal'].isin(['sell','SHORT','SHORT_ENTRY','exit_long','enter_short','exit_short','close_long'])]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys['datetime'], y=buys['price'], mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=10, name='Buy'))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells['datetime'], y=sells['price'], mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=10, name='Sell'))
    # vertical line for current bar
    if idx < len(df_plot):
        fig.add_vline(x=df_plot['datetime'].iloc[-1], line=dict(color='blue', width=1))
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # metrics
    latest_price = df_plot['close'].iloc[-1]
    eq = current_equity(latest_price)
    pnl = eq - INITIAL_EQUITY
    eq_text.metric("Equity", f"${eq:,.2f}")
    pl_text.metric("Unrealized P/L", f"${pnl:,.2f}")
    pct_text.metric("% Gain/Loss", f"{(pnl/INITIAL_EQUITY*100):.2f}%")

# Controls: Step / Play / Pause / Reset
controls_col1, controls_col2, controls_col3, controls_col4 = st.columns(4)
with controls_col1:
    if st.button("Step Forward"):
        st.session_state['playing'] = False
        if st.session_state['current_index'] < n_bars-1:
            st.session_state['current_index'] += 1
            i = st.session_state['current_index']
            # evaluate bar
            df_slice = data.iloc[:i+1]
            bar = data.iloc[i]
            sig = get_signal_from_module(strategy_module, df_slice, bar)
            # interpret signals (flexible mapping)
            if sig is not None:
                sig_s = str(sig).lower()
                if sig_s in ['buy','long','long_entry','enter_long']:
                    # position sizing
                    if fixed_size_units > 0:
                        size = fixed_size_units
                    else:
                        size = (st.session_state['cash'] * risk_pct) / bar['close']
                    enter_position(bar['close'], size)
                    signals_df.loc[len(signals_df)] = [bar['datetime'], 'buy', bar['close']]
                elif sig_s in ['sell','short','sell_entry','enter_short']:
                    # if we have position, exit it
                    if st.session_state['position']>0:
                        exit_position(bar['close'])
                        signals_df.loc[len(signals_df)] = [bar['datetime'], 'sell', bar['close']]
            update_display(st.session_state['current_index'])
with controls_col2:
    if st.button("Play / Resume"):
        st.session_state['playing'] = True
with controls_col3:
    if st.button("Pause"):
        st.session_state['playing'] = False
with controls_col4:
    if st.button("Reset"):
        st.session_state['current_index'] = 0
        st.session_state['cash'] = INITIAL_EQUITY
        st.session_state['position'] = 0.0
        st.session_state['signals'] = pd.DataFrame(columns=['datetime','signal','price'])
        st.session_state['trade_log'] = []
        st.session_state['playing'] = False
        update_display(0)

# Auto-play loop (non-blocking via session state)
if st.session_state['playing']:
    # iterate until end or until paused
    while st.session_state['playing'] and st.session_state['current_index'] < n_bars-1:
        st.session_state['current_index'] += 1
        i = st.session_state['current_index']
        df_slice = data.iloc[:i+1]
        bar = data.iloc[i]
        sig = get_signal_from_module(strategy_module, df_slice, bar)
        if sig is not None:
            sig_s = str(sig).lower()
            if sig_s in ['buy','long','long_entry','enter_long']:
                if fixed_size_units > 0:
                    size = fixed_size_units
                else:
                    size = (st.session_state['cash'] * risk_pct) / bar['close']
                enter_position(bar['close'], size)
                signals_df.loc[len(signals_df)] = [bar['datetime'], 'buy', bar['close']]
            elif sig_s in ['sell','short','sell_entry','enter_short']:
                if st.session_state['position']>0:
                    exit_position(bar['close'])
                    signals_df.loc[len(signals_df)] = [bar['datetime'], 'sell', bar['close']]
        update_display(st.session_state['current_index'])
        # sleep based on speed
        time.sleep(1.0 / max(1.0, speed))

# finally show trade log
with log_placeholder:
    st.subheader("Trade Log (most recent 20)")
    if len(st.session_state['trade_log'])>0:
        st.table(pd.DataFrame(st.session_state['trade_log']) .tail(20))
    else:
        st.write("No trades executed yet.")

# Batch Mode
st.sidebar.header("Batch / Reports")
if st.sidebar.button("Run Full Batch (simulate to end)"):
    # quick batch run without replotting each bar (faster)
    batch_df = data.copy()
    cash_b = INITIAL_EQUITY
    pos_b = 0.0
    signals_batch = []
    for i in range(len(batch_df)):
        bar = batch_df.iloc[i]
        sig = get_signal_from_module(strategy_module, batch_df.iloc[:i+1], bar)
        if sig is not None:
            s = str(sig).lower()
            if s in ['buy','long','enter_long']:
                if fixed_size_units > 0:
                    size = fixed_size_units
                else:
                    size = (cash_b * risk_pct) / bar['close']
                pos_b += size
                cash_b -= size*bar['close']
                signals_batch.append((bar['datetime'],'buy',bar['close']))
            elif s in ['sell','short','enter_short']:
                if pos_b>0:
                    cash_b += pos_b * bar['close']
                    pos_b = 0
                    signals_batch.append((bar['datetime'],'sell',bar['close']))
    final_equity = cash_b + pos_b * batch_df['close'].iloc[-1]
    st.sidebar.success(f"Batch complete. Final equity: ${final_equity:,.2f}")

# Live feed placeholder
st.sidebar.header("Live Feed")
if st.sidebar.checkbox("Enable live feed placeholder"):
    st.sidebar.info("This is a placeholder. Add code here to connect to broker/marketfeed and append rows to the data source.")
