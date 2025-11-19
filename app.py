import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# --- 頁面設定 ---
st.set_page_config(page_title="AI 美股預測實驗室", layout="wide")
st.title("🤖 AI 美股趨勢預測 (Prophet 模型)")
st.markdown("結合 **即時報價** 與 **Meta Prophet AI 模型**，推算未來可能的股價區間。")

# --- 側邊欄設定 ---
st.sidebar.header("設定參數")
ticker_input = st.sidebar.text_input("輸入美股代碼", value="NVDA")
forecast_days = st.sidebar.slider("AI 預測天數", min_value=30, max_value=365, value=90)

# --- 資料獲取函數 (絕對修正版) ---
@st.cache_data
def get_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        # 只抓取歷史數據
        hist = stock.history(period="5y")
        hist.reset_index(inplace=True)
        
        # 處理時區問題
        if 'Date' in hist.columns:
             hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        # 【關鍵修正】只回傳 hist (數據表)，不要回傳 stock
        return hist
    except Exception as e:
        return None

# --- AI 預測函數 ---
def predict_stock(data, days):
    df_train = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return m, forecast

# --- 主程式邏輯 ---
if ticker_input:
    ticker_symbol = ticker_input.upper()
    
    with st.spinner('正在下載數據並進行 AI 運算...'):
        # 【關鍵修正】這裡變成只接收一個變數 hist
        hist = get_stock_data(ticker_symbol)

        if hist is None or hist.empty:
            st.error("找不到代碼，請檢查拼字。")
        else:
            # 1. 顯示即時資訊
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            delta = current_price - prev_price
            delta_percent = (delta / prev_price) * 100
            
            st.subheader(f"目前股價: {ticker_symbol}")
            col1, col2, col3 = st.columns(3)
            col1.metric("最新收盤價", f"${current_price:.2f}", f"{delta:.2f} ({delta_percent:.2f}%)")
            col2.metric("預測天數", f"{forecast_days} 天")
            col3.metric("資料來源", "Yahoo Finance")
            
            st.divider()

            # 2. AI 預測圖表
            st.subheader(f"🔮 未來 {forecast_days} 天股價走勢預測")
            
            try:
                m, forecast = predict_stock(hist, forecast_days)
                fig = plot_plotly(m, forecast)
                fig.update_layout(
                    title=f"{ticker_symbol} 歷史數據與 AI 預測趨勢",
                    xaxis_title="日期",
                    yaxis_title="股價 (USD)",
                    hovermode="x",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 數值表
                st.subheader("📅 預測數值表 (未來 5 天)")
                future_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(5)
                future_data.columns = ['日期', '預測價格', '預測下限', '預測上限']
                st.dataframe(future_data.style.format({"預測價格": "{:.2f}", "預測下限": "{:.2f}", "預測上限": "{:.2f}"}))
                
            except Exception as e:
                st.error(f"預測模型運算錯誤: {e}")
