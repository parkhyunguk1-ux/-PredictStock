from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from agents.data_agent import fetch_stock_data, fetch_current_price, fetch_news
from agents.ml_agent import predict_price
from agents.llm_agent import analyze_sentiment

load_dotenv()

app = FastAPI(
    title="Stock Prediction Agent",
    description="하이브리드 주식 예측 Agent (ML + Claude LLM)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PriceResponse(BaseModel):
    ticker: str
    price: float
    change_pct: float
    volume: int
    timestamp: str


class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    expected_change_pct: float
    days_ahead: int
    direction_accuracy_pct: float
    investment_opinion: str
    sentiment_analysis: str
    news_count: int


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "Stock Prediction Agent"}


@app.get("/api/price/{ticker}", response_model=PriceResponse, tags=["Market Data"])
def get_price(ticker: str):
    try:
        return fetch_current_price(ticker.upper())
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/predict/{ticker}", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    ticker: str,
    days: int = Query(default=5, ge=1, le=30),
):
    ticker = ticker.upper()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    try:
        df = fetch_stock_data(ticker, period="6mo")
        price_data = fetch_current_price(ticker)
        news = fetch_news(ticker, max_items=5)
        ml_result = predict_price(df, days_ahead=days)
        llm_result = analyze_sentiment(ticker, news, price_data, ml_result)

        return PredictionResponse(
            ticker=ticker,
            current_price=ml_result["current_price"],
            predicted_price=ml_result["predicted_price"],
            expected_change_pct=ml_result["expected_change_pct"],
            days_ahead=days,
            direction_accuracy_pct=ml_result["direction_accuracy_pct"],
            investment_opinion=llm_result["investment_opinion"],
            sentiment_analysis=llm_result["sentiment_analysis"],
            news_count=len(news),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")


@app.get("/api/news/{ticker}", tags=["Market Data"])
def get_news(ticker: str, limit: int = Query(default=5, ge=1, le=20)):
    try:
        news = fetch_news(ticker.upper(), max_items=limit)
        return {"ticker": ticker.upper(), "count": len(news), "news": news}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 정적 파일 마운트는 항상 맨 마지막에
app.mount("/", StaticFiles(directory="static", html=True), name="static")
