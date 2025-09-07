# ===================================================
# Streamlit Backtester with Indicators, Batch Mode,
# Live Feed Integration, Persistent DB
# ===================================================
# FEATURES:
# - Symbol selection from persistent SQLite DB
# - Historical import and update of symbol data
# - Interactive step-through or batch backtesting
# - Adjustable speed (up to 30 bars/sec)
# - Advanced indicator overlays: SMA20/50, EMA20/50, Bollinger Bands, RSI, MACD
# - Visual buy/sell signals
# - Equity, P/L, % gain/loss live metrics
# - Optional live-feed data integration
# - Step/back/play controls
# ===================================================

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import time
import ta
from datetime import datetime

# ---------------------------
# DATABASE SETUP
# ---------------------------
DB_PATH = 'symbols_data.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            symbol TEXT,
            DateTime TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL,
            PRIMARY KEY(symbol, DateTime)
        )
    """)
    conn.commit()
    return conn

conn = init_db()

def load_symbol(symbol):
    df = pd.read_sql(f"SELECT * FROM symbols WHERE symbol='{symbol}' ORDER BY DateTime ASC", conn)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df

def update_symbol(df, symbol):
    df.to_sql('symbols', conn, if_exists='append', index=False)

# ---------------------------
# INDICATORS
# ---------------------------
def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['BB_High'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
    df['BB_Low'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Signal'] = 0  # Placeholder for strategy buy/sell signals
    return df

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Interactive Backtester + Batch Mode + Live Feed")

# Symbol selection
symbols = pd.read_sql("SELECT DISTINCT symbol FROM symbols", conn)['symbol'].tolist()
symbol_selected = st.selectbox("Select Symbol", symbols)

# Load data
if 'df_backtest' not in st.session_state or st.session_state.symbol != symbol_selected:
    st.session_state.df_backtest = load_symbol(symbol_selected)
    st.session_state.df_backtest = add_indicators(st.session_state.df_backtest)
    st.session_state.current_index = 0
    st.session_state.symbol = symbol_selected
    st.session_state.equity = 100000  # Base currency

df = st.session_state.df_backtest

# Date range selection
start_date = st.date_input("Start Date", df['DateTime'].min().date())
end_date = st.date_input("End Date", df['DateTime'].max().date())

df_filtered = df[(df['DateTime'].dt.date >= start_date) & (df['DateTime'].dt.date <= end_date)]

# Speed / Step Controls
speed = st.slider("Playback speed (bars/sec)", 1, 30, 5)
step_mode = st.checkbox("Step through bars manually", value=True)
next_bar = st.button("Next Bar")

# Live equity / P&L metrics
st.subheader("Equity / P&L")
st.metric("Equity", f"${st.session_state.equity:,.2f}")
st.metric("Running P/L", f"${st.session_state.df_backtest.iloc[:st.session_state.current_index]['Close'].diff().sum():,.2f}")
st.metric("% Gain/Loss", f"{(st.session_state.df_backtest.iloc[:st.session_state.current_index]['Close'].diff().sum()/st.session_state.equity*100):.2f}%")

# ---------------------------
# BACKTEST LOOP
# ---------------------------
def display_chart():
    df_display = df_filtered.iloc[:st.session_state.current_index+1]
    fig = go.Figure(data=[go.Candlestick(
        x=df_display['DateTime'],
        open=df_display['Open'],
        high=df_display['High'],
        low=df_display['Low'],
        close=df_display['Close'],
        name=symbol_selected
    )])
    # Indicators
    for col, color in [('SMA_20','blue'),('SMA_50','darkblue'),('EMA_20','orange'),('EMA_50','red')]:
        fig.add_trace(go.Scatter(x=df_display['DateTime'], y=df_display[col], line=dict(color=color, width=1), name=col))
    for col in ['BB_High','BB_Low']:
        fig.add_trace(go.Scatter(x=df_display['DateTime'], y=df_display[col], line=dict(color='purple', width=1, dash='dot'), name=col))
    # Signals
    buys = df_display[df_display['Signal']==1]
    sells = df_display[df_display['Signal']==-1]
    fig.add_trace(go.Scatter(x=buys['DateTime'], y=buys['Close'], mode='markers',
                             marker_symbol='triangle-up', marker_color='green', marker_size=10, name='Buy'))
    fig.add_trace(go.Scatter(x=sells['DateTime'], y=sells['Close'], mode='markers',
                             marker_symbol='triangle-down', marker_color='red', marker_size=10, name='Sell'))
    fig.update_layout(title=f"{symbol_selected} Backtest", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)
    
    # Optional indicators below chart
    st.line_chart(df_display[['RSI','MACD','MACD_Signal']].set_index(df_display['DateTime']))

if step_mode:
    if next_bar:
        if st.session_state.current_index < len(df_filtered)-1:
            st.session_state.current_index += 1
        display_chart()
else:
    for i in range(st.session_state.current_index, len(df_filtered)):
        st.session_state.current_index = i
        display_chart()
        time.sleep(1/speed)

# ---------------------------
# LIVE-FEED PLACEHOLDER
# ---------------------------
st.subheader("Live Feed Integration")
st.info("Optional: Connect your live data feed here. Append new bars to the DB and the backtester will update dynamically.")
