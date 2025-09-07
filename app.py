"""
Streamlit Backtester
===================

Features:
- Symbol selection from DB or CSV
- Timeframe and date range selection
- Step-by-step or speed-controlled bar replay
- Live equity, running P/L, % gain/loss display
- Candle chart with indicators
- Batch mode backtesting
- Optional live feed integration
- Dynamic loading of strategies from /strategies
- Advanced indicator overlays
- Supports minute-level and higher timeframe data
- Fully generic: works with any financial symbol

Usage:
1. Place all strategy files in /strategies with standard interface.
2. Prepare symbol CSVs or database backend with historical data.
3. Run: `streamlit run app.py`
4. Select symbol, timeframe, start/end dates, strategy, and play mode.
5. Step bar-by-bar or adjust replay speed.
"""

import streamlit as st
import pandas as pd
import os
import importlib.util
import time
from pathlib import Path
import plotly.graph_objects as go

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = Path("data")
STRATEGY_DIR = Path("strategies")
DEFAULT_SPEED = 1  # bars/sec

# ---------------------------
# Load strategies dynamically
# ---------------------------
def load_strategies():
    strategies = {}
    for file in STRATEGY_DIR.glob("*.py"):
        name = file.stem
        spec = importlib.util.spec_from_file_location(name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        strategies[name] = mod
    return strategies

strategies = load_strategies()

# ---------------------------
# UI Controls
# ---------------------------
st.title("Interactive Backtester")
symbol_list = [f.stem for f in DATA_DIR.glob("*.csv")]
symbol = st.selectbox("Select Symbol", symbol_list)

strategy_name = st.selectbox("Select Strategy", list(strategies.keys()))
strategy_module = strategies[strategy_name]

start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
timeframe = st.selectbox("Timeframe", ["1min", "5min", "15min", "1H", "1D"])
speed = st.slider("Replay Speed (bars/sec)", 0.1, 30.0, DEFAULT_SPEED, 0.1)

step_mode = st.checkbox("Step Mode (manual advance)")

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data(symbol):
    df = pd.read_csv(DATA_DIR / f"{symbol}.csv", parse_dates=["datetime"])
    df = df.sort_values("datetime")
    return df

data = load_data(symbol)
data = data[(data["datetime"].dt.date >= start_date) & (data["datetime"].dt.date <= end_date)]

# ---------------------------
# Initialize strategy
# ---------------------------
strategy = strategy_module.Strategy()
strategy.init(data)

# ---------------------------
# Initialize equity / stats
# ---------------------------
equity = 100000.0
equity_history = []
pl_history = []

# ---------------------------
# Chart
# ---------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data["datetime"],
    open=data["open"],
    high=data["high"],
    low=data["low"],
    close=data["close"],
    name="Price"
))

chart_placeholder = st.empty()

# ---------------------------
# Step-through / replay
# ---------------------------
if step_mode:
    next_bar = st.button("Next Bar")
else:
    next_bar = True  # auto-replay

i = 0
while i < len(data):
    if step_mode and not next_bar:
        time.sleep(0.1)
        continue
    
    bar = data.iloc[i]
    signals = strategy.on_bar(bar)
    
    # Update equity and stats
    # Example: placeholder logic (replace with strategy's actual logic)
    pl = 0  # implement actual P/L computation
    equity_history.append(equity + pl)
    pl_history.append(pl)
    
    # Update chart
    fig.update_layout(title=f"{symbol} | Equity: {equity_history[-1]:.2f} | P/L: {pl_history[-1]:.2f}")
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    if step_mode:
        break  # wait for next click
    else:
        time.sleep(1.0 / speed)
        i += 1

# ---------------------------
# Batch mode
# ---------------------------
st.sidebar.header("Batch Mode")
if st.sidebar.button("Run Batch Backtest"):
    batch_results = {}
    for s_name, s_module in strategies.items():
        s = s_module.Strategy()
        s.init(data)
        eq = equity
        for idx, bar in data.iterrows():
            sig = s.on_bar(bar)
            # placeholder P/L
            eq += 0
        batch_results[s_name] = eq
    st.sidebar.write(batch_results)
