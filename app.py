import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# --- 頁面設定 (手機版優化) ---
st.set_page_config(page_title="AI 美股預測", layout="wide")

# --- 標題區 ---
st.title("🤖 AI 美股預測")
st.caption("輸入代碼 (如 NVDA, TSLA, AAPL) 查看即時走勢與 AI 預測")

# --- 輸入區 (搬到主畫面，方便手機輸入) ---
col_input, col_days = st.columns([2, 1])

with col_input:
    # 這裡就是您要的輸入欄位，預設 NVDA
    ticker_input = st.text_input("請輸入美股代碼", value="NVDA")

with col_days:
    # 預測天數設定
    forecast_days = st.selectbox("預測天數", [30, 60, 90, 180, 365], index=2)

# --- 資料獲取函數 (增強版：雙重保險) ---
@st.cache_data
def get_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # 嘗試 1: 抓取自動調整後的股價 (解決分割問題)
        hist = stock.history(period="5y", auto_adjust=True)
        
        # 如果抓不到 (例如新上市股票或 API 異常)，嘗試 2: 抓原始股價
        if hist is None or hist.empty:
            hist = stock.history(period="5y", auto_adjust=False)
        
        # 如果還是空的，那就是真的代碼錯了
        if hist is None or hist.empty:
            return None

        hist.reset_index(inplace=True)
        
        # 處理時區問題
        if 'Date' in hist.columns:
             hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        return hist
    except Exception as e:
        return None

# --- AI 預測函數 ---
def predict_stock(data, days):
    df_train = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # 針對個股優化的參數
    m = Prophet(daily_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return m, forecast

# --- 主程式邏輯 ---
if ticker_input:
    ticker_symbol = ticker_input.upper().strip() # 去除前後空白
    
    # 顯示讀取動畫
    with st.spinner(f'正在分析 {ticker_symbol} 的數據...'):
        hist = get_stock_data(ticker_symbol)

        if hist is None or hist.empty:
            st.error(f"❌ 找不到代碼 '{ticker_symbol}'。")
            st.info("💡 提示：美股代碼通常是英文縮寫，例如台積電請輸入 TSM。")
        else:
            # 1. 顯示即時資訊 (卡片式設計)
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            delta = current_price - prev_price
            delta_percent = (delta / prev_price) * 100
            
            # 根據漲跌變色 (美股：綠漲紅跌)
            color = "green" if delta >= 0 else "red"
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 20px;">
                <h2 style="margin:0; color: white;">{ticker_symbol}</h2>
                <h1 style="margin:0; color: {color};">${current_price:.2f}</h1>
                <p style="margin:0; color: {color};">{delta:+.2f} ({delta_percent:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            # 2. AI 預測圖表
            st.subheader(f"📈 趨勢預測 ({forecast_days}天)")
            
            try:
                m, forecast = predict_stock(hist, forecast_days)
                fig = plot_plotly(m, forecast)
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="股價 (USD)",
                    hovermode="x",
                    height=500, # 手機版高度稍微調小一點
                    margin=dict(l=20, r=20, t=40, b=20) # 調整邊界
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 未來價格表
                st.subheader("📅 未來 5 天預測")
                last_hist_date = hist['Date'].iloc[-1]
                future_only = forecast[forecast['ds'] > last_hist_date]
                future_data = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(5)
                
                future_data.columns = ['日期', '預測價', '下限', '上限']
                future_data['日期'] = future_data['日期'].dt.strftime('%m-%d') # 手機版日期簡化為 月-日
                
                # 顯示表格
                st.dataframe(
                    future_data.style.format({"預測價": "{:.1f}", "下限": "{:.1f}", "上限": "{:.1f}"}),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"分析失敗: {e}")
