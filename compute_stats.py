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
