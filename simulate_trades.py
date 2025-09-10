def simulate_trades(df, trades, sl_pct=0.01, tp_pct=0.02, initial_balance=100000, risk_per_trade=0.01):
    """
    Given raw trade signals (Date, Direction, Price),
    simulate SL/TP and update equity curve.
    """
    balance = initial_balance
    balances = []
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
