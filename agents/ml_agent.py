import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["RSI"] = _rsi(df["Close"], 14)
    df["Price_Change"] = df["Close"].pct_change()
    df["Volatility"] = df["Close"].rolling(5).std()
    df["Volume_MA"] = df["Volume"].rolling(5).mean()
    return df.dropna()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def predict_price(df: pd.DataFrame, days_ahead: int = 5) -> dict:
    df = add_features(df)

    feature_cols = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA20", "RSI", "Price_Change", "Volatility", "Volume_MA"]
    target_col = "Close"

    # 미래 N일 후 종가를 타깃으로 사용
    df["Target"] = df[target_col].shift(-days_ahead)
    df = df.dropna()

    X = df[feature_cols].values
    y = df["Target"].values

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 마지막 20%는 테스트
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 가장 최근 데이터로 예측
    last_features = scaler_X.transform([df[feature_cols].values[-1]])
    predicted_price = float(model.predict(last_features)[0])
    current_price = float(df["Close"].iloc[-1])

    # 테스트셋 정확도
    test_preds = model.predict(X_test)
    mae = float(np.mean(np.abs(test_preds - y_test)))
    direction_acc = float(np.mean(np.sign(test_preds - current_price) == np.sign(y_test - current_price)) * 100)

    return {
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "expected_change_pct": round((predicted_price - current_price) / current_price * 100, 2),
        "days_ahead": days_ahead,
        "mae": round(mae, 2),
        "direction_accuracy_pct": round(direction_acc, 1),
    }
