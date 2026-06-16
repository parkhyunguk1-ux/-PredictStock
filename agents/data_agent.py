import urllib3
urllib3.disable_warnings()

import pandas as pd
import requests
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com",
}


def _yahoo_chart(ticker: str, range_: str = "6mo", interval: str = "1d") -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": range_, "interval": interval, "includePrePost": "false"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"'{ticker}' 데이터를 가져올 수 없습니다. 티커를 확인해 주세요.")
    return result[0]


def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    range_map = {"6mo": "6mo", "1y": "1y", "2y": "2y"}
    range_ = range_map.get(period, "6mo")

    result = None
    for r in [range_, "1y", "2y"]:
        try:
            result = _yahoo_chart(ticker, range_=r)
            if result.get("timestamp"):
                break
        except Exception:
            continue

    if not result or not result.get("timestamp"):
        raise ValueError(f"'{ticker}' 데이터를 가져올 수 없습니다. 티커를 확인해 주세요.")

    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Open": quotes.get("open", []),
        "High": quotes.get("high", []),
        "Low": quotes.get("low", []),
        "Close": quotes.get("close", []),
        "Volume": quotes.get("volume", []),
    }, index=pd.to_datetime(timestamps, unit="s"))

    return df.dropna()


def fetch_current_price(ticker: str) -> dict:
    result = _yahoo_chart(ticker, range_="5d", interval="1d")

    quotes = result["indicators"]["quote"][0]
    closes = [c for c in quotes.get("close", []) if c is not None]

    if not closes:
        raise ValueError(f"'{ticker}' 현재가를 가져올 수 없습니다.")

    current = closes[-1]
    prev = closes[-2] if len(closes) >= 2 else current
    change_pct = (current - prev) / prev * 100

    volumes = [v for v in quotes.get("volume", []) if v is not None]
    volume = int(volumes[-1]) if volumes else 0

    return {
        "ticker": ticker.upper(),
        "price": round(current, 2),
        "change_pct": round(change_pct, 2),
        "volume": volume,
        "timestamp": datetime.now().isoformat(),
    }


def fetch_news(ticker: str, max_items: int = 5) -> list[dict]:
    import feedparser
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        return [
            {
                "title": e.get("title", ""),
                "summary": e.get("summary", ""),
                "published": e.get("published", ""),
            }
            for e in feed.entries[:max_items]
        ]
    except Exception:
        return []
