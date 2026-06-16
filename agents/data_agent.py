import urllib3
urllib3.disable_warnings()

import yfinance as yf
import pandas as pd
from datetime import datetime
import requests

def _make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
    })
    return session


def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    session = _make_session()
    stock = yf.Ticker(ticker, session=session)

    df = pd.DataFrame()
    for p in [period, "1y", "2y", "max"]:
        try:
            df = stock.history(period=p)
            if not df.empty:
                break
        except Exception:
            continue

    if df.empty:
        raise ValueError(f"'{ticker}' 데이터를 가져올 수 없습니다. 티커를 확인하거나 잠시 후 다시 시도해 주세요.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def fetch_current_price(ticker: str) -> dict:
    session = _make_session()
    stock = yf.Ticker(ticker, session=session)

    hist = pd.DataFrame()
    for p in ["5d", "1mo"]:
        try:
            hist = stock.history(period=p)
            hist = hist.dropna(subset=["Close"])
            if not hist.empty:
                break
        except Exception:
            continue

    if hist.empty:
        raise ValueError(f"'{ticker}' 현재가를 가져올 수 없습니다.")

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
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        news = []
        for entry in feed.entries[:max_items]:
            news.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
            })
        return news
    except Exception:
        return []
