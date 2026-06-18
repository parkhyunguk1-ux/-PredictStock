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
    df["Volume_MA5"] = df["Volume"].rolling(5).mean()
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA5"] + 1)  # 평균 대비 거래량 비율
    df["Price_Volume"] = df["Close"] * df["Volume"]  # 거래대금
    return df.dropna()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _train_model(X_scaled, y, feature_cols, df):
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    return model, X_test, y_test


def predict_price(df: pd.DataFrame, days_ahead: int = 5) -> dict:
    df = add_features(df)

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "MA5", "MA20", "RSI", "Price_Change", "Volatility",
        "Volume_MA5", "Volume_MA20", "Volume_Change", "Volume_Ratio", "Price_Volume"
    ]

    # ── 가격 예측 모델 ──
    df_price = df.copy()
    df_price["Target"] = df_price["Close"].shift(-days_ahead)
    df_price = df_price.dropna()

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df_price[feature_cols].values)
    y_price = df_price["Target"].values

    price_model, X_test_p, y_test_p = _train_model(X_scaled, y_price, feature_cols, df_price)
    last_features = scaler_X.transform([df_price[feature_cols].values[-1]])
    predicted_price = float(price_model.predict(last_features)[0])
    current_price = float(df_price["Close"].iloc[-1])

    test_preds_p = price_model.predict(X_test_p)
    direction_acc = float(np.mean(
        np.sign(test_preds_p - current_price) == np.sign(y_test_p - current_price)
    ) * 100)

    # ── 거래량 예측 모델 ──
    df_vol = df.copy()
    df_vol["Target"] = df_vol["Volume"].shift(-days_ahead)
    df_vol = df_vol.dropna()

    scaler_V = MinMaxScaler()
    XV_scaled = scaler_V.fit_transform(df_vol[feature_cols].values)
    y_volume = df_vol["Target"].values

    vol_model, X_test_v, y_test_v = _train_model(XV_scaled, y_volume, feature_cols, df_vol)
    last_vol_features = scaler_V.transform([df_vol[feature_cols].values[-1]])
    predicted_volume = float(vol_model.predict(last_vol_features)[0])
    current_volume = float(df_vol["Volume"].iloc[-1])

    vol_change_pct = (predicted_volume - current_volume) / (current_volume + 1) * 100

    # 거래량 신호: 가격 상승 + 거래량 증가 = 강한 매수, 가격 하락 + 거래량 증가 = 강한 매도
    price_up = predicted_price > current_price
    vol_up = predicted_volume > current_volume

    if price_up and vol_up:
        volume_signal = "🔥 강한 매수 신호 (가격↑ + 거래량↑)"
        signal_color = "green"
    elif price_up and not vol_up:
        volume_signal = "📈 약한 매수 신호 (가격↑ + 거래량↓)"
        signal_color = "lightgreen"
    elif not price_up and vol_up:
        volume_signal = "⚠️ 강한 매도 신호 (가격↓ + 거래량↑)"
        signal_color = "red"
    else:
        volume_signal = "📉 약한 매도 신호 (가격↓ + 거래량↓)"
        signal_color = "orange"

    return {
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "expected_change_pct": round((predicted_price - current_price) / current_price * 100, 2),
        "days_ahead": days_ahead,
        "direction_accuracy_pct": round(direction_acc, 1),
        "current_volume": int(current_volume),
        "predicted_volume": int(max(predicted_volume, 0)),
        "volume_change_pct": round(vol_change_pct, 2),
        "volume_signal": volume_signal,
        "signal_color": signal_color,
    }
