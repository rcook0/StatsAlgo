import streamlit as st
import pandas as pd
import numpy as np
import os
import importlib.util
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go

# =========================
# Backtester Features
# =========================
# 1. Dynamic symbol selection from DB/CSV
# 2. Date/time range picker based on historical data
# 3. Timeframe selection (1m, 5m, 15m, 1h)
# 4. Strategy loader from 'strategies/' folder
# 5. Step-through/backtesting controls
# 6. Candlestick chart updating per step
# 7. Live equity, running P/L, % gain/loss
# 8. Trade log display
# =========================

# -------------------------
# Load available symbols
# -------------------------
symbol_files = [f for f in os.listdir('data') if f.endswith('.csv')]
symbol = st.selectbox("Select Symbol", symbol_files)

# Load CSV
data = pd.read_csv(f'data/{symbol}', parse_dates=['datetime'])
data.sort_values('datetime', inplace=True)

# Date/time selection
start_dt = st.date_input('Start Date', value=data['datetime'].min().date())
end_dt = st.date_input('End Date', value=data['datetime'].max().date())
filtered_data = data[(data['datetime'].dt.date >= start_dt) & (data['datetime'].dt.date <= end_dt)]

# Timeframe selection
timeframe = st.selectbox('Timeframe', ['1min', '5min', '15min', '1h'])

# -------------------------
# Load strategies dynamically
# -------------------------
strategy_files = [f for f in os.listdir('strategies') if f.endswith('.py')]
strategy_name = st.selectbox('Select Strategy', strategy_files)

spec = importlib.util.spec_from_file_location('strategy_module', f'strategies/{strategy_name}')
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)
strategy_class = getattr(strategy_module, 'Strategy')

# -------------------------
# Initialize backtesting
# -------------------------
initial_equity = st.number_input('Initial Equity', value=10000.0)
equity = initial_equity
trade_log = []

# Step-through controls
step_speed = st.slider('Bars per Second', 1, 30, 5)
manual_step = st.button('Step')

# Candlestick chart setup
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=filtered_data['datetime'],
    open=filtered_data['open'],
    high=filtered_data['high'],
    low=filtered_data['low'],
    close=filtered_data['close'],
    name='Price'
))
fig.update_layout(xaxis_rangeslider_visible=False)

# -------------------------
# Backtest loop
# -------------------------
placeholder_chart = st.empty()
placeholder_eq = st.empty()
placeholder_log = st.empty()

for i in range(1, len(filtered_data)):
    bar = filtered_data.iloc[i]
    strategy = strategy_class(filtered_data.iloc[:i+1])
    signal = strategy.get_signal()  # 'buy', 'sell', 'hold'

    if signal == 'buy':
        trade_log.append({'datetime': bar['datetime'], 'action': 'buy', 'price': bar['close']})
        # Simple P/L simulation
        equity -= bar['close']
    elif signal == 'sell':
        trade_log.append({'datetime': bar['datetime'], 'action': 'sell', 'price': bar['close']})
        equity += bar['close']

    # Update visuals
    placeholder_chart.plotly_chart(fig)
    placeholder_eq.metric('Equity', f'{equity:.2f}', f'{((equity - initial_equity)/initial_equity)*100:.2f}%')
    placeholder_log.dataframe(pd.DataFrame(trade_log))

    # Handle step or speed
    if manual_step:
        st.stop()
    else:
        time.sleep(1/step_speed)
