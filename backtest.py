import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate intraday 1-min data for 3 months trading days (approx 60 days)
np.random.seed(42)

def simulate_us30_intraday(start_date='2025-01-01', days=60):
    minutes_per_day = 390  # US market open 9:30 to 16:00 = 6.5 hours * 60
    total_minutes = minutes_per_day * days

    # Simulate price returns ~ N(0, 0.0005) per minute (~0.05% std)
    returns = np.random.normal(0, 0.0005, total_minutes)

    # Build datetime index skipping weekends
    all_dates = pd.bdate_range(start_date, periods=days)
    datetimes = []
    for day in all_dates:
        day_times = pd.date_range(day.strftime('%Y-%m-%d') + ' 09:30', periods=minutes_per_day, freq='T')
        datetimes.extend(day_times)
    datetimes = pd.to_datetime(datetimes)

    prices = 40000 * (1 + returns).cumprod()  # Start near 40000 (typical DJ level)
    df = pd.DataFrame({'datetime': datetimes, 'return': returns, 'price': prices})
    df['open'] = df['price'].shift(1)
    df['open'].iloc[0] = df['price'].iloc[0]
    df['close'] = df['price']
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.0002, len(df)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.0002, len(df)))

    return df.drop(columns=['price', 'return'])

df = simulate_us30_intraday()

# 2. Calculate open-to-15min returns at NY open (09:30)
def calculate_open_to_15min_return(df):
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time

    results = []
    unique_dates = df['date'].unique()

    for day in unique_dates:
        day_data = df[df['date'] == day]
        open_bar = day_data[day_data['time'] == pd.to_datetime('09:30').time()]
        if open_bar.empty:
            continue
        open_price = open_bar['open'].values[0]

        # 15 minutes later index (assume 1-min bars)
        idx = open_bar.index[0]
        idx_15min = idx + 15
        if idx_15min >= day_data.index[-1]:
            continue
        price_15min = df.loc[idx_15min, 'close']

        ret = (price_15min - open_price) / open_price
        results.append({'date': day, 'open_price': open_price, 'price_15min': price_15min, 'return_15min': ret})

    return pd.DataFrame(results)

returns_df = calculate_open_to_15min_return(df)

# 3. Calculate rolling mean/std for open-to-15min returns (lookback 20 days)
lookback = 20
returns_df['mean_ret'] = returns_df['return_15min'].rolling(lookback).mean()
returns_df['std_ret'] = returns_df['return_15min'].rolling(lookback).std()

# 4. Volatility filter: Use std dev of open-to-15min returns over lookback window
returns_df['volatility'] = returns_df['return_15min'].rolling(lookback).std()
vol_threshold = returns_df['volatility'].median()

# 5. Trading logic with adaptive threshold and fixed holding period exit (30 mins after 15min mark)
threshold_sigma = 2

# We need price at entry (open+15min) and exit (open+45min)
def get_price_at_time(df, day, minutes_after_open):
    day_data = df[df['datetime'].dt.date == day]
    open_time = pd.to_datetime(str(day) + ' 09:30')
    target_time = open_time + pd.Timedelta(minutes=minutes_after_open)
    row = day_data[day_data['datetime'] == target_time]
    if row.empty:
        return None
    return row['close'].values[0]

trades = []

for i in range(lookback, len(returns_df)):
    row = returns_df.iloc[i]
    if pd.isna(row['mean_ret']) or pd.isna(row['std_ret']) or pd.isna(row['volatility']):
        continue

    # Volatility filter
    if row['volatility'] < vol_threshold:
        continue

    ret = row['return_15min']
    mean = row['mean_ret']
    std = row['std_ret']

    # Adaptive threshold example: increase threshold if volatility high
    adaptive_threshold = threshold_sigma
    if row['volatility'] > vol_threshold * 1.5:
        adaptive_threshold = threshold_sigma * 1.2

    signal = 0
    if ret > mean + adaptive_threshold * std:
        signal = -1  # short
    elif ret < mean - adaptive_threshold * std:
        signal = 1   # long

    if signal != 0:
        entry_price = row['price_15min']
        exit_price = get_price_at_time(df, row['date'], 45)  # exit 30 min after entry (15 + 30 = 45)
        if exit_price is None:
            continue

        # Calculate return from entry to exit
        trade_return = (exit_price - entry_price) / entry_price * signal  # long=+, short=-
        trades.append({'date': row['date'], 'signal': signal, 'entry_price': entry_price,
                       'exit_price': exit_price, 'return': trade_return})

trades_df = pd.DataFrame(trades)

# 6. Performance stats
win_rate = (trades_df['return'] > 0).mean()
avg_win = trades_df.loc[trades_df['return'] > 0, 'return'].mean()
avg_loss = trades_df.loc[trades_df['return'] <= 0, 'return'].mean()
total_return = trades_df['return'].sum()
num_trades = len(trades_df)

print(f"Trades taken: {num_trades}")
print(f"Win rate: {win_rate:.2%}")
print(f"Avg win: {avg_win:.4f}, Avg loss: {avg_loss:.4f}")
print(f"Total return (sum of trade returns): {total_return:.4f}")

# 7. Kelly Position Sizing Explanation & Calculation
#
# Kelly fraction: K = W - (1 - W) / R
# where W = win rate, R = avg win / abs(avg loss)

R = avg_win / abs(avg_loss) if avg_loss != 0 else 0
W = win_rate

if R > 0:
    kelly_fraction = W - (1 - W) / R
    kelly_fraction = max(kelly_fraction, 0)  # no negative sizing
else:
    kelly_fraction = 0

print(f"Kelly fraction: {kelly_fraction:.2%}")

# 8. Plot cumulative returns of strategy (assume equal sizing)
trades_df['cum_return'] = trades_df['return'].cumsum()
plt.plot(trades_df['date'], trades_df['cum_return'])
plt.title('Cumulative Strategy Return (Simulated Data)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

