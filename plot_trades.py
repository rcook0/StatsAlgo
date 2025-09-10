import plotly.graph_objects as go

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
