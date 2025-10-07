import yfinance as yf
import pandas as pd
from typing import Optional
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from textblob import TextBlob
from rag.vector_store import StockVectorStore


def stock_price_chart(symbol: str, period: str = "1mo", interval: str = "1d") -> go.Figure:
    """Return a Plotly OHLC chart for the given symbol."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No price data for {symbol}")
        return fig

    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name=symbol
    )])
    fig.update_layout(title=f"{symbol} Price ({period})", xaxis_title="Date", yaxis_title="Price")
    return fig


def sentiment_trend_chart(vector_store: StockVectorStore, symbol: str, days_back: int = 14) -> go.Figure:
    """Plot average daily sentiment from documents in the vector store for the symbol."""
    # Retrieve recent docs for symbol with an explicit query
    results = vector_store.search_by_symbol(symbol, query="recent stock analysis news sentiment", top_k=200)
    if not results:
        fig = go.Figure()
        fig.update_layout(title=f"No documents for {symbol}")
        return fig

    rows = []
    for r in results:
        # Extract content and timestamp from result structure
        content = r.get("content") or r.get("preview", "")
        meta = r.get("metadata", {})
        ts = meta.get("timestamp")
        if not ts:
            continue
        try:
            polarity = TextBlob(content).sentiment.polarity
        except Exception:
            polarity = 0.0
        # Robust timestamp parsing (handle ISO strings with/without Z)
        try:
            ts_parsed = pd.to_datetime(str(ts).replace('Z', '+00:00'))
        except Exception:
            ts_parsed = pd.to_datetime(ts, errors='coerce')
        if pd.isna(ts_parsed):
            continue
        rows.append({"date": ts_parsed.date(), "sentiment": polarity})

    if not rows:
        fig = go.Figure()
        fig.update_layout(title=f"No sentiment data for {symbol}")
        return fig

    df = pd.DataFrame(rows)
    cutoff = pd.Timestamp.today().date() - pd.Timedelta(days=days_back)
    df = df[df["date"] >= cutoff]
    daily = df.groupby("date").sentiment.mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["sentiment"], mode="lines+markers", name="Sentiment"))
    fig.update_layout(title=f"{symbol} Sentiment Trend ({days_back}d)", xaxis_title="Date", yaxis_title="Avg Sentiment")
    return fig


def combined_insights_figure(vector_store: StockVectorStore, symbol: str, period: str = "1mo", interval: str = "1d") -> go.Figure:
    """Two-panel figure: price chart and sentiment trend."""
    price_fig = stock_price_chart(symbol, period=period, interval=interval)
    sent_fig = sentiment_trend_chart(vector_store, symbol)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12, subplot_titles=("Price", "Sentiment"))

    # Price (Candlestick)
    for trace in price_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # Sentiment trend
    for trace in sent_fig.data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(height=800, title_text=f"{symbol} Insights")
    return fig