import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 股市預測 v8.1", layout="wide")
st.title("🤖 AI 股市預測 v8.1")
st.caption("介面優化版：主畫面直接切換市場 (美股/台股)")

# --- 2. 市場與輸入設定 (移至主畫面) ---
# 使用 horizontal=True 讓選項橫向排列，手機點選更直覺
market_mode = st.radio("", ["🇺🇸 美股 (US)", "🇹🇼 台股 (TW)"], horizontal=True)

col_input, col_days = st.columns([2, 1])

with col_input:
    if market_mode == "🇺🇸 美股 (US)":
        default_ticker = "NVDA"
        label_text = "輸入代碼 (如 NVDA)"
        currency = "USD"
        currency_symbol = "$"
        placeholder = "NVDA"
    else:
        default_ticker = "2330"
        label_text = "輸入代碼 (如 2330)"
        currency = "TWD"
        currency_symbol = "NT$"
        placeholder = "2330"
        
    ticker_input = st.text_input(label_text, value=default_ticker, placeholder=placeholder)

with col_days:
    forecast_days = st.selectbox("預測天數", [30, 60, 90, 180], index=1)

# --- 3. 資料獲取函數 ---
@st.cache_data
def get_stock_data(ticker, market):
    try:
        # 台股邏輯：自動偵測 .TW 或 .TWO
        if market == "🇹🇼 台股 (TW)":
            if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
                test_ticker = f"{ticker}.TW"
            else:
                test_ticker = ticker
            
            stock = yf.Ticker(test_ticker)
            hist = stock.history(period="5y", auto_adjust=True)
            
            if hist is None or hist.empty:
                test_ticker = f"{ticker}.TWO"
                stock = yf.Ticker(test_ticker)
                hist = stock.history(period="5y", auto_adjust=True)
        else:
            # 美股邏輯
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y", auto_adjust=True)

        if hist is None or hist.empty:
            hist = stock.history(period="5y", auto_adjust=False)
        
        if hist is None or hist.empty:
            return None, None, None

        hist.reset_index(inplace=True)
        if 'Date' in hist.columns:
             hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        info = stock.info
        real_symbol = stock.ticker 
        return hist, info, real_symbol

    except Exception:
        return None, None, None

# --- 4. AI 預測函數 ---
def predict_stock(data, days):
    df_train = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=days, freq='B')
    forecast = m.predict(future)
    
    cols_to_fix = ['yhat', 'yhat_lower', 'yhat_upper']
    forecast[cols_to_fix] = forecast[cols_to_fix].clip(lower=0)
    return m, forecast

# --- 5. 回測函數 ---
def backtest_model(data, test_days=5):
    df_full = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    # 確保資料足夠進行切割
    if len(df_full) < test_days + 30:
        return 0, pd.DataFrame()
        
    train_df = df_full.iloc[:-test_days]
    test_df = df_full.iloc[-test_days:].copy()
    
    m = Prophet(daily_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=test_days, freq='B')
    forecast = m.predict(future)
    
    forecast_tail = forecast.tail(test_days)[['ds', 'yhat']]
    result = pd.merge(test_df, forecast_tail, on='ds', how='inner')
    
    if result.empty:
        return 0, pd.DataFrame()

    result['error_pct'] = ((result['y'] - result['yhat']).abs() / result['y']) * 100
    acc_score = 100 - result['error_pct'].mean()
    return acc_score, result

# --- 6. 繪圖輔助函數 ---
def plot_gauge(current, future, c_symbol):
    change_pct = ((future - current) / current) * 100
    if change_pct >= 10: rating, color = "強烈買進", "#00CC96"
    elif change_pct >= 5: rating, color = "買進", "#2ca02c"
    elif change_pct > -5: rating, color = "持守", "#ffbf00"
    elif change_pct > -10: rating, color = "賣出", "#d62728"
    else: rating, color = "強烈賣出", "#8c1515"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = change_pct,
        title = {'text': f"AI 建議: {rating}", 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'color': color}, 'valueformat': "+.1f"},
        gauge = {
            'axis': {'range': [-30, 30]}, 'bar': {'color': "white"}, 'bgcolor': "black",
            'steps': [
                {'range': [-30, -10], 'color': '#8c1515'}, {'range': [-10, -5], 'color': '#d62728'},
                {'range': [-5, 5], 'color': '#ffbf00'}, {'range': [5, 10], 'color': '#2
