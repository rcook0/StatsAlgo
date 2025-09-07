import streamlit as st
import pandas as pd
import os, importlib.util
import time

# -----------------------
# Load data
# -----------------------
df = pd.read_csv('data/US30_1min.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# -----------------------
# Load strategies
# -----------------------
STRATEGY_DIR = 'strategies'

def load_strategies():
    strategies = {}
    if not os.path.exists(STRATEGY_DIR):
        os.makedirs(STRATEGY_DIR)
    for file in os.listdir(STRATEGY_DIR):
        if file.endswith('.py'):
            path = os.path.join(STRATEGY_DIR, file)
            spec = importlib.util.spec_from_file_location(file[:-3], path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, 'generate_signal'):
                strategies[file[:-3]] = mod.generate_signal
    return strategies

strategies = load_strategies()
strategy_selected = st.selectbox("Select Strategy", list(strategies.keys()))
strategy_func = strategies[strategy_selected]

# -----------------------
# Session state
# -----------------------
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'df_backtest' not in st.session_state:
    st.session_state.df_backtest = df.copy()
    st.session_state.df_backtest['Signal'] = 0

# -----------------------
# Step controls
# -----------------------
col1, col2, col3 = st.columns(3)
if col1.button("Step Back"):
    st.session_state.current_index = max(0, st.session_state.current_index - 1)
if col2.button("Step Forward"):
    st.session_state.current_index = min(len(df)-1, st.session_state.current_index + 1)
speed = col3.slider("Speed (bars/sec)", 1, 30, 5)

# -----------------------
# Generate signals
# -----------------------
df_display = df.iloc[:st.session_state.current_index+1].copy()
if len(df_display) > 0:
    df_display['Signal'] = df_display.apply(lambda row: strategy_func(df_display.loc[:row.name]), axis=1)
    st.session_state.df_backtest.loc[df_display.index, 'Signal'] = df_display['Signal']

# -----------------------
# Plot
# -----------------------
st.line_chart(df_display[['Close']])
st.write(df_display.tail(10))

# -----------------------
# Auto-play
# -----------------------
if st.button("Run Auto"):
    for _ in range(st.session_state.current_index, len(df)):
        st.session_state.current_index += 1
        time.sleep(1/speed)
        st.experimental_rerun()
