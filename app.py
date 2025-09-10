# app.py
"""
Streamlit Statistical Backtester
--------------------------------
Features:
- Upload CSV or generate demo OHLCV data
- 15+ statistical/probabilistic strategies (imported from /strategies)
- Dropdown menu to select strategy
- SL/TP enforcement with risk-based position sizing
- Interactive candlestick chart with trade entries (LONG/SHORT)
- Equity curve visualization
- Summary stats table: trades, win rate, avg pnl, max DD, final equity, % gain
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ------------------------
# Demo Data Generator
# ------------------------
def generate_demo_data(rows=500, start_price=35000, seed=42):
    np.random.seed(seed)
    dates = [datetime(2023, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(rows)]
    prices = [start_price]

    for _ in range(1, rows):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.0007)))  # ~0.07% std

    df = pd.DataFrame({"Date": dates})
    df["Close"] = prices
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + np.random.uniform(0, 0.001, rows))
    df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - np.random.uniform(0, 0.001, rows))
    df["Volume"] = np.random.randint(100, 5000, size=rows)
    return df

# ------------------------
# Trade Plotting Function
# ------------------------
def plot_trades(df, trades):
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    )])

    longs = [(t[0], t[2]) for t in trades if t[1] == "LONG"]
    shorts = [(t[0], t[2]) for t in trades if t[1] == "SHORT"]

    if longs:
        fig.add_trace(go.Scatter(
            x=[x for x, _ in longs],
            y=[y for _, y in longs],
            mode="markers",
            name="Long Entry",
            marker=dict(color="green", size=8, symbol="triangle-up")
        ))

    if shorts:
        fig.add_trace(go.Scatter(
            x=[x for x, _ in shorts],
            y=[y for _, y in shorts],
            mode="markers",
            name="Short Entry",
            marker=dict(color="red", size=8, symbol="triangle-down")
        ))

    fig.update_layout(title="Strategy Trades", xaxis_rangeslider_visible=False)
    return fig

# ------------------------
# Equity Simulation with SL/TP
# ------------------------
def simulate_trades(df, trades, sl_pct=0.01, tp_pct=0.02, initial_balance=100000, risk_per_trade=0.01):
    balance = initial_balance
    equity_curve = []
    detailed_trades = []

    for trade in trades:
        entry_time, direction, entry_price = trade
        size = balance * risk_per_trade / (sl_pct * entry_price)  # position sizing
        sl = entry_price * (1 - sl_pct if direction == "LONG" else 1 + sl_pct)
        tp = entry_price * (1 + tp_pct if direction == "LONG" else 1 - tp_pct)

        exit_price = entry_price
        exit_time = entry_time
        pnl = 0

        # Walk forward until SL or TP hit
        trade_df = df[df["Date"] >= entry_time]
        for _, row in trade_df.iterrows():
            high, low, close, dt = row["High"], row["Low"], row["Close"], row["Date"]

            if direction == "LONG":
                if low <= sl:
                    exit_price, exit_time = sl, dt
                    pnl = -sl_pct * balance * risk_per_trade
                    break
                elif high >= tp:
                    exit_price, exit_time = tp, dt
                    pnl = tp_pct * balance * risk_per_trade
                    break
            else:  # SHORT
                if high >= sl:
                    exit_price, exit_time = sl, dt
                    pnl = -sl_pct * balance * risk_per_trade
                    break
                elif low <= tp:
                    exit_price, exit_time = tp, dt
                    pnl = tp_pct * balance * risk_per_trade
                    break

        balance += pnl
        detailed_trades.append({
            "EntryTime": entry_time,
            "ExitTime": exit_time,
            "Direction": direction,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "PnL": pnl
        })
        equity_curve.append((exit_time, balance))

    eq_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"])
    trades_df = pd.DataFrame(detailed_trades)
    return eq_df, trades_df

def plot_equity(eq_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df["Date"],
        y=eq_df["Equity"],
        mode="lines",
        name="Equity"
    ))
    fig.update_layout(title="Equity Curve")
    return fig

# ------------------------
# Stats Table
# ------------------------
def compute_stats(trades_df, eq_df, initial_balance=100000):
    total_trades = len(trades_df)
    wins = trades_df[trades_df["PnL"] > 0]
    losses = trades_df[trades_df["PnL"] <= 0]

    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = trades_df["PnL"].mean() if total_trades > 0 else 0
    final_equity = eq_df["Equity"].iloc[-1] if not eq_df.empty else initial_balance
    pct_gain = (final_equity / initial_balance - 1) * 100

    # Max drawdown
    roll_max = eq_df["Equity"].cummax()
    drawdown = (eq_df["Equity"] - roll_max) / roll_max
    max_dd = drawdown.min() * 100 if not eq_df.empty else 0

    stats = {
        "Total Trades": total_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Avg PnL": round(avg_pnl, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Final Equity": round(final_equity, 2),
        "Net % Gain": round(pct_gain, 2)
    }
    return stats

# ------------------------
# Strategy Loader
# ------------------------
strategy_files = {
    "Sigma Extreme": "sigma_extreme",
    "Opening Range Breakout": "opening_range_breakout",
    "Gap Fade/Continuation": "gap_fade",
    "Volatility Breakout": "volatility_breakout",
    "Time-of-Day Mean Reversion": "time_of_day_mean_reversion",
    "Microstructure Imbalance": "microstructure_imbalance",
    "Mean Reversion on Returns": "mean_reversion_returns",
    "Probabilistic Time Edge": "probabilistic_time_edge",
    "Overnight Gap": "overnight_gap",
    "Sign Sequence Probabilities": "sign_sequence_prob",
    "Return Clustering": "return_clustering",
    "Extreme Quantile": "extreme_quantile",
    "Expected Value Maximization": "expected_value",
    "Sequential Reversal": "sequential_reversal",
    "Monte Carlo Bootstrapping": "monte_carlo_bootstrap",
    "Custom Placeholder": "custom_placeholder"
}

def load_strategy(strategy_name, df):
    module_name = strategy_files[strategy_name]
    module = importlib.import_module(f"strategies.{module_name}")
    strategy_class = getattr(module, "".join([w.capitalize() for w in module_name.split("_")]))
    return strategy_class(df)

# ------------------------
# Streamlit App
# ------------------------
st.title("📊 Statistical Backtester")

# Sidebar Data Load
st.sidebar.header("Data Loader")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_demo = st.sidebar.button("Load Demo Data")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.success("✅ CSV loaded")
elif use_demo:
    df = generate_demo_data()
    st.success("✅ Demo data generated")
else:
    df = None
    st.info("Upload a CSV or click 'Load Demo Data'")

# Strategy selection
st.sidebar.header("Strategy")
strategy_name = st.sidebar.selectbox("Choose a strategy", list(strategy_files.keys()))

# SL/TP settings
st.sidebar.header("Risk Settings")
sl_pct = st.sidebar.number_input("Stop Loss %", 0.001, 0.1, 0.01, 0.001)
tp_pct = st.sidebar.number_input("Take Profit %", 0.001, 0.2, 0.02, 0.001)
risk_per_trade = st.sidebar.number_input("Risk per Trade (fraction)", 0.001, 0.05, 0.01, 0.001)

if df is not None:
    if st.sidebar.button("Run Backtest"):
        st.subheader(f"Running Strategy: {strategy_name}")
        strategy = load_strategy(strategy_name, df)
        trades = strategy.run()

        st.write(f"Number of raw signals: {len(trades)}")
        eq_df, trades_df = simulate_trades(df, trades, sl_pct, tp_pct, risk_per_trade=risk_per_trade)

        st.plotly_chart(plot_trades(df, trades), use_container_width=True)
        st.plotly_chart(plot_equity(eq_df), use_container_width=True)

        stats = compute_stats(trades_df, eq_df)
        st.subheader("📊 Strategy Performance Summary")
        st.table(pd.DataFrame([stats]))
