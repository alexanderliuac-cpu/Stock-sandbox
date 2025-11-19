import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# --- 頁面設定 ---
st.set_page_config(page_title="AI 美股預測", layout="wide")

# --- 標題區 ---
st.title("🤖 AI 美股預測")
st.caption("v4.0: 負值校正 (No Negative Price) & 排除週末")

# --- 輸入區 ---
col_input, col_days = st.columns([2, 1])

with col_input:
    ticker_input = st.text_input("請輸入美股代碼", value="NVAX")

with col_days:
    forecast_days = st.selectbox("預測範圍", [30, 60, 90, 180], index=1)

# --- 資料獲取函數 ---
@st.cache_data
def get_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # 嘗試 1: 抓取自動調整後的股價
        hist = stock.history(period="5y", auto_adjust=True)
        
        # 嘗試 2: 抓原始股價
        if hist is None or hist.empty:
            hist = stock.history(period="5y", auto_adjust=False)
        
        if hist is None or hist.empty:
            return None

        hist.reset_index(inplace=True)
        
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
    
    # 設定未來預測僅含工作日 (Business Day)
    future = m.make_future_dataframe(periods=days, freq='B')
    
    forecast = m.predict(future)
    
    # 【v4.0 關鍵修正】: 將所有預測價格 (yhat) 與區間 (lower/upper) 強制鎖在 0 以上
    cols_to_fix = ['yhat', 'yhat_lower', 'yhat_upper']
    forecast[cols_to_fix] = forecast[cols_to_fix].clip(lower=0)
    
    return m, forecast

# --- 主程式邏輯 ---
if ticker_input:
    ticker_symbol = ticker_input.upper().strip()
    
    with st.spinner(f'正在分析 {ticker_symbol}...'):
        hist = get_stock_data(ticker_symbol)

        if hist is None or hist.empty:
            st.error(f"❌ 找不到代碼 '{ticker_symbol}'。")
        else:
            # 1. 即時資訊卡片
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            delta = current_price - prev_price
            delta_percent = (delta / prev_price) * 100
            color = "green" if delta >= 0 else "red"
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 20px;">
                <h2 style="margin:0; color: white;">{ticker_symbol}</h2>
                <h1 style="margin:0; color: {color};">${current_price:.2f}</h1>
                <p style="margin:0; color: {color};">{delta:+.2f} ({delta_percent:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            # 2. AI 預測圖表
            st.subheader(f"📈 趨勢預測 ({forecast_days}個交易日)")
            
            try:
                m, forecast = predict_stock(hist, forecast_days)
                fig = plot_plotly(m, forecast)
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="股價 (USD)",
                    hovermode="x",
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 未來 10 天預測表
                st.subheader("📅 未來 10 個交易日預測")
                
                last_hist_date = hist['Date'].iloc[-1]
                future_only = forecast[forecast['ds'] > last_hist_date]
                
                future_data = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10)
                
                future_data.columns = ['日期 (週一至週五)', '預測價', '下限', '上限']
                future_data['日期 (週一至週五)'] = future_data['日期 (週一至週五)'].dt.strftime('%m-%d (%a)')
                
                st.dataframe(
                    future_data.style.format({"預測價": "{:.2f}", "下限": "{:.2f}", "上限": "{:.2f}"}),
                    use_container_width=True,
                    height=400
                )
                
            except Exception as e:
                st.error(f"分析失敗: {e}")
