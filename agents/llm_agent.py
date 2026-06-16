import os
import anthropic


def analyze_sentiment(ticker: str, news: list[dict], price_data: dict, ml_prediction: dict) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    news_text = "\n".join(
        f"- {n['title']}: {n['summary'][:200]}" for n in news
    ) if news else "뉴스 없음"

    prompt = f"""당신은 전문 주식 애널리스트입니다. 다음 정보를 바탕으로 {ticker} 주식에 대한 투자 분석 리포트를 작성하세요.

## 현재 주가 정보
- 현재가: ${price_data['price']}
- 전일 대비 변동: {price_data['change_pct']}%

## ML 모델 예측 ({ml_prediction['days_ahead']}일 후)
- 예측 가격: ${ml_prediction['predicted_price']}
- 예상 변동률: {ml_prediction['expected_change_pct']}%
- 방향 정확도(백테스트): {ml_prediction['direction_accuracy_pct']}%

## 최근 뉴스
{news_text}

다음 형식으로 한국어로 분석해주세요:

### 감성 분석
뉴스와 시장 분위기에 대한 종합 평가 (긍정/중립/부정)

### 투자 의견
매수 / 보유 / 매도 중 하나와 그 근거

### 리스크 요인
주요 리스크 2-3가지

### 종합 요약
2-3문장으로 핵심 요약

⚠️ 이 분석은 참고용이며, 실제 투자 결정은 본인 책임입니다."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    analysis_text = message.content[0].text

    # 투자 의견 파싱
    opinion = "보유"
    if "매수" in analysis_text:
        opinion = "매수"
    elif "매도" in analysis_text:
        opinion = "매도"

    return {
        "sentiment_analysis": analysis_text,
        "investment_opinion": opinion,
        "model_used": "claude-sonnet-4-6",
    }
