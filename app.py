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

# --- 資料獲取函數 (加入快取以提升效能) ---
@st.cache_data
def get_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        # 抓取過去 5 年資料來訓練模型
        hist = stock.history(period="5y")
        hist.reset_index(inplace=True)
        
        # 處理時區問題 (Prophet 不喜歡時區資訊)
        hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        return stock, hist
    except Exception as e:
        return None, None

# --- AI 預測函數 ---
def predict_stock(data, days):
    # 準備 Prophet 需要的格式: ds (時間), y (數值)
    df_train = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # 建立並訓練模型
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    
    # 建立未來日期的 DataFrame
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    
    return m, forecast

# --- 主程式邏輯 ---
if ticker_input:
    ticker_symbol = ticker_input.upper()
    
    with st.spinner('正在下載數據並進行 AI 運算...'):
        stock, hist = get_stock_data(ticker_symbol)

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
            
            # 執行預測
            m, forecast = predict_stock(hist, forecast_days)
            
            # 使用 Prophet 內建的 Plotly 繪圖功能
            fig = plot_plotly(m, forecast)
            
            # 優化圖表外觀
            fig.update_layout(
                title=f"{ticker_symbol} 歷史數據與 AI 預測趨勢",
                xaxis_title="日期",
                yaxis_title="股價 (USD)",
                hovermode="x",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # 3. 預測數據解讀
            with st.expander("📊 如何解讀這張圖？ (點擊展開)"):
                st.markdown("""
                * **黑點 (Black Dots)**: 實際的歷史股價數據。
                * **深藍線 (Blue Line)**: AI 認為的「最主要趨勢」。
                * **淺藍色區域 (Light Blue Area)**: 這是**不確定性區間 (Confidence Interval)**。
                    * AI 表示：「我有 80% 的信心，未來的股價會落在這個淺藍色範圍內。」
                    * 如果淺藍色區域越寬，代表波動越大，預測越不準確。
                """)

            # 4. 顯示具體預測數值 (最後 5 天)
            st.subheader("📅 預測數值表 (未來 5 天)")
            future_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
            future_data.columns = ['日期', '預測價格', '預測下限', '預測上限']
            st.dataframe(future_data.style.format({"預測價格": "{:.2f}", "預測下限": "{:.2f}", "預測上限": "{:.2f}"}))

            st.warning("⚠️ 免責聲明：此模型僅基於歷史數據進行數學統計推算，無法預測突發新聞、政策變動或黑天鵝事件。請勿僅依賴此工具進行投資決策。")
