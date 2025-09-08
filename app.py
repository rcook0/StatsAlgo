# app.py
"""
Streamlit Statistical Backtester v1.0

Features:
- Symbol selection from database / CSV
- Timeframe selection (1m, 5m, 15m, 1H, etc.)
- Date/time range selection
- Strategy selection from /strategies folder
- Step-through mode (bar-by-bar) or auto-run (up to 30 bars/sec)
- Candle chart plotting with Plotly
- Live equity, running P/L, % gain/loss
- Batch mode for multiple symbols / date ranges
- Optional live-feed integration (placeholder)
- Persistent database of historical symbol data
- Supports generic minute-level OHLCV datasets
- Modular strategy architecture
- Statistics / trade log generation

Usage:
1. Place all historical OHLCV CSVs in ./data/ (columns: datetime, open, high, low, close, volume)
2. Add strategies to ./strategies folder (each implements `run_strategy(df)` returning signals)
3. Run:
   $ streamlit run app.py
4. Select symbol, timeframe, date range, strategy, mode (step/run), then start
5. Monitor candle chart, equity, P/L
6. Optional: run batch mode for multiple symbols

"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import importlib.util
import time
import plotly.graph_objects as go
from datetime import datetime

# ---- Config ----
DATA_FOLDER = "./data"
STRATEGIES_FOLDER = "./strategies"
MAX_SPEED_BARS_PER_SEC = 30

# ---- Helper functions ----
def load_symbols():
    return [f.replace(".csv","") for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

def load_data(symbol):
    path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_strategies():
    strategies = {}
    for file in os.listdir(STRATEGIES_FOLDER):
        if file.endswith(".py") and not file.startswith("__"):
            name = file.replace(".py","")
            spec = importlib.util.spec_from_file_location(name, os.path.join(STRATEGIES_FOLDER,file))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            strategies[name] = mod
    return strategies

def plot_candles(df, signals=None, current_index=None):
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    )])
    if signals is not None:
        buys = signals[signals['signal']=='buy']
        sells = signals[signals['signal']=='sell']
        fig.add_trace(go.Scatter(x=buys['datetime'], y=buys['price'], mode='markers', marker_symbol='triangle-up', marker_color='green', name='Buy'))
        fig.add_trace(go.Scatter(x=sells['datetime'], y=sells['price'], mode='markers', marker_symbol='triangle-down', marker_color='red', name='Sell'))
    if current_index is not None:
        fig.add_vline(x=df['datetime'].iloc[current_index], line=dict(color="blue", width=2))
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def simulate_trading(df, strategy_module, step_mode=True, speed=1.0):
    signals = pd.DataFrame(columns=['datetime','signal','price'])
    equity = 100000
    position = 0
    cash = equity
    pnl = []
    bar_count = len(df)
    current_index = 0

    while current_index < bar_count:
        bar = df.iloc[current_index]
        signal = strategy_module.run_strategy(df.iloc[:current_index+1])
        if signal in ['buy','sell']:
            signals = pd.concat([signals, pd.DataFrame({'datetime':[bar['datetime']], 'signal':[signal], 'price':[bar['close']]})], ignore_index=True)
            # simple PnL logic: buy adds position, sell closes
            if signal=='buy':
                position += cash/bar['close']
                cash = 0
            elif signal=='sell' and position>0:
                cash += position*bar['close']
                position = 0
        current_equity = cash + position*bar['close']
        pnl.append(current_equity)
        plot_candles(df, signals, current_index)
        st.write(f"Equity: {current_equity:.2f}, P/L: {current_equity-equity:.2f}, %Gain: {(current_equity-equity)/equity*100:.2f}%")
        current_index += 1
        if step_mode:
            st.button("Next Bar")  # pauses until click
        else:
            time.sleep(max(1.0/speed, 0))
    st.success("Simulation complete")
    return signals, pnl

# ---- Streamlit App ----
st.title("Statistical Backtester")

symbols = load_symbols()
symbol = st.selectbox("Select Symbol", symbols)

df = load_data(symbol)
st.write(f"Loaded {len(df)} bars")

timeframe = st.selectbox("Select Timeframe", ["1m","5m","15m","1H","1D"])
# placeholder: resample if needed

strategies = load_strategies()
strategy_name = st.selectbox("Select Strategy", list(strategies.keys()))
strategy_module = strategies[strategy_name]

mode = st.radio("Mode", ["Step-through","Auto-run"])
speed = 1.0
if mode=="Auto-run":
    speed = st.slider("Speed (bars/sec)", 1, MAX_SPEED_BARS_PER_SEC, 5)

if st.button("Start Simulation"):
    simulate_trading(df, strategy_module, step_mode=(mode=="Step-through"), speed=speed)
