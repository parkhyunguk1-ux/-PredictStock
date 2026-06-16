import ssl
import urllib3
urllib3.disable_warnings()

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"티커 '{ticker}'에 대한 데이터를 찾을 수 없습니다.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def fetch_current_price(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.fast_info
    hist = stock.history(period="5d")
    hist = hist.dropna(subset=["Close"])

    current = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
    change_pct = (current - prev) / prev * 100

    return {
        "ticker": ticker.upper(),
        "price": round(current, 2),
        "change_pct": round(change_pct, 2),
        "volume": int(hist["Volume"].iloc[-1]),
        "timestamp": datetime.now().isoformat(),
    }


def fetch_news(ticker: str, max_items: int = 5) -> list[dict]:
    import feedparser

    company_name = ticker  # 간단히 티커 사용
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)

    news = []
    for entry in feed.entries[:max_items]:
        news.append({
            "title": entry.get("title", ""),
            "summary": entry.get("summary", ""),
            "published": entry.get("published", ""),
        })
    return news
