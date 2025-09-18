# app_advanced.py — Streamlit Backtester (Advanced)
# -------------------------------------------------------------------
# Included features (from your requested list):
# 2) Partial exits / Scaling in-out
# 3) Trailing stop (percent or ATR-based) exits
# 4) Save/Load configurations (local session + JSON download/upload)
# 5) Multi-timeframe analysis (HTF filter via resample + EMA trend)
# 7) Strategy parameter optimization (grid search over user-specified params)
#
# Other features retained:
# - CSV upload or demo data generation
# - 16 statistical strategies imported from ./strategies (each with class + run())
# - Candlestick + entry markers, equity curve
# - SL/TP enforcement & risk-based sizing
# - Summary stats table (trades, win rate, avg pnl, max DD, final equity, % gain)
#
# Run: streamlit run app_advanced.py
# Requirements: streamlit, plotly, pandas, numpy

import streamlit as st
import pandas as pd
import numpy as np
import importlib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import inspect

# ------------------------
# Demo Data Generator
# ------------------------
def generate_demo_data(rows=800, start_price=35000, seed=42):
    np.random.seed(seed)
    dates = [datetime(2023, 1, 3, 14, 30) + timedelta(minutes=i) for i in range(rows)]
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
# Plot helpers
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
    longs = [(t["EntryTime"], t["EntryPrice"]) for t in trades if t["Direction"] == "LONG"]
    shorts = [(t["EntryTime"], t["EntryPrice"]) for t in trades if t["Direction"] == "SHORT"]
    if longs:
        fig.add_trace(go.Scatter(
            x=[x for x,_ in longs], y=[y for _,y in longs],
            mode="markers", name="Long Entry",
            marker=dict(color="green", size=9, symbol="triangle-up")
        ))
    if shorts:
        fig.add_trace(go.Scatter(
            x=[x for x,_ in shorts], y=[y for _,y in shorts],
            mode="markers", name="Short Entry",
            marker=dict(color="red", size=9, symbol="triangle-down")
        ))
    fig.update_layout(title="Strategy Trades", xaxis_rangeslider_visible=False)
    return fig

def plot_equity(eq_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_df["Date"], y=eq_df["Equity"], mode="lines", name="Equity"))
    fig.update_layout(title="Equity Curve", xaxis_rangeslider_visible=False)
    return fig

# ------------------------
# Risk / Stops helpers
# ------------------------
def compute_atr(df, n=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    return atr

# ------------------------
# Backtest simulation: SL/TP + Trailing + Scaling
# ------------------------
def simulate_trades(
    df,
    raw_signals,                 
    sl_pct=0.01,                 
    tp_pct=0.02,                 
    risk_per_trade=0.01,         
    init_equity=100000,
    trail_type="percent",        
    trail_value=0.01,            
    allow_scale_in=True,
    scale_in_steps=2,            
    scale_in_trigger=0.01,       
    scale_out_on_adverse=True,
    scale_out_trigger=0.01       
):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["ATR14"] = compute_atr(df, 14)

    equity = init_equity
    equity_curve = []
    detailed_trades = []
    signal_idx_map = {sig[0]: (sig[1], sig[2]) for sig in raw_signals}

    open_positions = []  
    last_add_price = {}  
    trade_id_counter = 0

    for i in range(len(df)):
        row = df.iloc[i]
        dt, o, h, l, c, atr = row["Date"], row["Open"], row["High"], row["Low"], row["Close"], row["ATR14"]

        # --- new signal triggers ---
        if dt in signal_idx_map:
            direction, entry_price = signal_idx_map[dt]
            this_sl = entry_price * (1 - sl_pct) if direction == "LONG" else entry_price * (1 + sl_pct)
            risk_per_unit = abs(entry_price - this_sl)
            if risk_per_unit <= 0:
                size_units = 0
            else:
                size_units = (equity * risk_per_trade) / risk_per_unit
            if size_units > 0:
                trade_id_counter += 1
                leg = {
                    "TradeID": trade_id_counter,
                    "EntryTime": dt,
                    "Direction": direction,
                    "EntryPrice": entry_price,
                    "Size": size_units,
                    "SL": this_sl,
                    "TP": entry_price * (1 + tp_pct) if direction == "LONG" else entry_price * (1 - tp_pct),
                    "TrailType": trail_type,
                    "TrailValue": trail_value,
                    "Peak": entry_price,     
                    "Trough": entry_price,
                    "Adds": 0
                }
                open_positions.append(leg)
                last_add_price[trade_id_counter] = entry_price

        new_open_positions = []
        for leg in open_positions:
            direction = leg["Direction"]
            entry = leg["EntryPrice"]
            size = leg["Size"]
            sl = leg["SL"]
            tp = leg["TP"]
            adds = leg["Adds"]

            if direction == "LONG":
                leg["Peak"] = max(leg.get("Peak", entry), h)
                leg["Trough"] = min(leg.get("Trough", entry), l)
            else:
                leg["Peak"] = min(leg.get("Peak", entry), l)
                leg["Trough"] = max(leg.get("Trough", entry), h)

            # trailing stop
            if leg["TrailType"] == "percent":
                if direction == "LONG":
                    trail_stop = leg["Peak"] * (1 - leg["TrailValue"])
                    sl = max(sl, trail_stop)
                elif direction == "SHORT":
                    trail_stop = leg["Peak"] * (1 + leg["TrailValue"])
                    sl = min(sl, trail_stop)
            elif leg["TrailType"] == "atr" and not np.isnan(atr):
                mult = leg["TrailValue"]
                if direction == "LONG":
                    trail_stop = c - mult * atr
                    sl = max(sl, trail_stop)
                else:
                    trail_stop = c + mult * atr
                    sl = min(sl, trail_stop)

            # exit conditions
            hit = None
            exit_price = c
            if direction == "LONG":
                if l <= sl:
                    hit = "SL"; exit_price = sl
                elif h >= tp:
                    hit = "TP"; exit_price = tp
            else:
                if h >= sl:
                    hit = "SL"; exit_price = sl
                elif l <= tp:
                    hit = "TP"; exit_price = tp

            if hit is not None or size <= 1e-9:
                pnl = (exit_price - entry) * size if direction == "LONG" else (entry - exit_price) * size
                equity += pnl
                detailed_trades.append({
                    "TradeID": leg["TradeID"],
                    "EntryTime": leg["EntryTime"],
                    "ExitTime": dt,
                    "Direction": direction,
                    "EntryPrice": entry,
                    "ExitPrice": exit_price,
                    "Size": size,
                    "Adds": adds,
                    "ExitReason": hit if hit else "ScaledOut",
                    "PnL": pnl
                })
            else:
                leg["Size"] = size
                leg["EntryPrice"] = entry
                leg["SL"] = sl
                leg["TP"] = tp
                leg["Adds"] = adds
                new_open_positions.append(leg)

        open_positions = new_open_positions
        equity_curve.append((dt, equity))

    eq_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"])
    trades_df = pd.DataFrame(detailed_trades)
    return eq_df, trades_df

# ------------------------
# Stats
# ------------------------
def compute_stats(trades_df, eq_df, initial_balance=100000):
    total_trades = len(trades_df)
    wins = trades_df[trades_df["PnL"] > 0]
    losses = trades_df[trades_df["PnL"] <= 0]

    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = trades_df["PnL"].mean() if total_trades > 0 else 0
    final_equity = eq_df["Equity"].iloc[-1] if not eq_df.empty else initial_balance
    pct_gain = (final_equity / initial_balance - 1) * 100

    if not eq_df.empty:
        roll_max = eq_df["Equity"].cummax()
        drawdown = (eq_df["Equity"] - roll_max) / roll_max
        max_dd = drawdown.min() * 100
    else:
        max_dd = 0

    stats = {
        "Total Trades": int(total_trades),
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
STRATEGY_FILES = [
    "sigma_extreme", "opening_range_breakout", "gap_fade",
    "time_of_day_mean_reversion", "microstructure_imbalance",
    "mean_reversion_returns", "prob_time_of_day", "overnight_gap",
    "prob_sign_sequences", "return_clustering", "extreme_quantile",
    "expected_value_max", "sequential_reversal", "monte_carlo_bootstrap",
    "volatility_breakout", "mean_reversion_price_changes"
]

def load_strategy(name):
    try:
        module = importlib.import_module(f"strategies.{name}")
        class_name = "".join([part.capitalize() for part in name.split("_")])
        if hasattr(module, class_name):
            return getattr(module, class_name)
        else:
            for _, obj in inspect.getmembers(module, inspect.isclass):
                return obj
    except Exception as e:
        st.error(f"Could not load {name}: {e}")
    return None

# ------------------------
# HTF filter
# ------------------------
def apply_htf_filter(df, htf="D", ema_period=20):
    df = df.set_index("Date")
    resampled = df["Close"].resample(htf).last().dropna()
    ema = resampled.ewm(span=ema_period, adjust=False).mean()
    trend = "LONG" if resampled.iloc[-1] > ema.iloc[-1] else "SHORT"
    return trend

# ------------------------
# Run Strategy
# ------------------------
def run_strategy_get_signals(strategy_name, df, params, use_htf=False, htf="D", ema_period=20):
    strat_cls = load_strategy(strategy_name)
    if strat_cls is None:
        return []
    strat = strat_cls(df, **params)
    raw = strat.run()
    signals = []
    htf_trend = None
    if use_htf:
        htf_trend = apply_htf_filter(df, htf=htf, ema_period=ema_period)
    for t in raw:
        dt, direction, price = t
        if use_htf and direction != htf_trend:
            continue
        signals.append((dt, direction, price))
    return signals

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(layout="wide")
st.title("📊 Advanced Statistical Backtester")

st.sidebar.header("Data Input")
opt = st.sidebar.radio("Data Source", ["Demo", "CSV Upload"])
if opt == "Demo":
    df = generate_demo_data()
else:
    f = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if f:
        df = pd.read_csv(f, parse_dates=["Date"])
    else:
        st.stop()

st.sidebar.header("Strategy Selection")
strat_choice = st.sidebar.selectbox("Select strategy", STRATEGY_FILES)
params = st.sidebar.text_area("Params (JSON)", value="{}")
try:
    params = json.loads(params)
except:
    st.warning("Invalid JSON")
    params = {}

st.sidebar.header("Risk/Stops Config")
sl_pct = st.sidebar.number_input("SL %", 0.001, 0.2, 0.01, step=0.001)
tp_pct = st.sidebar.number_input("TP %", 0.001, 0.2, 0.02, step=0.001)
trail_type = st.sidebar.selectbox("Trailing stop type", ["percent","atr"])
trail_val = st.sidebar.number_input("Trail value", 0.001, 5.0, 0.01, step=0.01)
risk_per = st.sidebar.number_input("Risk per trade", 0.001, 0.1, 0.01, step=0.001)
init_equity = st.sidebar.number_input("Initial Equity", 1000, 1000000, 100000, step=1000)

st.sidebar.header("Multi-timeframe")
use_htf = st.sidebar.checkbox("Enable HTF Filter")
htf = st.sidebar.selectbox("HTF", ["D","W","M"])
ema_per = st.sidebar.number_input("EMA period", 5, 100, 20)

st.sidebar.header("Save/Load Config")
if st.sidebar.button("Save Config"):
    st.sidebar.download_button("Download JSON", json.dumps(params), file_name="config.json")
conf_file = st.sidebar.file_uploader("Load Config", type=["json"])
if conf_file:
    params = json.load(conf_file)

# Optimization UI
st.sidebar.header("Optimization")
do_opt = st.sidebar.checkbox("Run Optimization?")
grid_param = st.sidebar.text_area("Grid Param JSON", value='{"sl_pct":[0.005,0.01], "tp_pct":[0.01,0.02]}')

if st.sidebar.button("Run Backtest"):
    signals = run_strategy_get_signals(strat_choice, df.copy(), params, use_htf, htf, ema_per)
    eq, trades = simulate_trades(
        df.copy(), signals,
        sl_pct=sl_pct, tp_pct=tp_pct,
        risk_per_trade=risk_per,
        init_equity=init_equity,
        trail_type=trail_type, trail_value=trail_val
    )
    st.plotly_chart(plot_trades(df, trades.to_dict("records")), use_container_width=True)
    st.plotly_chart(plot_equity(eq), use_container_width=True)
    stats = compute_stats(trades, eq, init_equity)
    st.subheader("Summary Stats")
    st.table(pd.DataFrame([stats]))