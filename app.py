import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- 頁面設定 ---
st.set_page_config(page_title="AI 美股預測 v5.0", layout="wide")

# --- 標題區 ---
st.title("🤖 AI 美股預測 v5.0")
st.caption("含建議光譜 (Recommendation Spectrum) & 負值校正 & 假日排除")

# --- 輸入區 ---
col_input, col_days = st.columns([2, 1])

with col_input:
    ticker_input = st.text_input("請輸入美股代碼", value="NVDA")

with col_days:
    # 這裡的選擇會直接影響 AI 對「買賣建議」的判斷基準
    forecast_days = st.selectbox("預測範圍", [30, 60, 90, 180], index=1)

# --- 資料獲取函數 ---
@st.cache_data
def get_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="5y", auto_adjust=True)
        
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
    
    m = Prophet(daily_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(df_train)
    
    future = m.make_future_dataframe(periods=days, freq='B')
    forecast = m.predict(future)
    
    # 負值校正
    cols_to_fix = ['yhat', 'yhat_lower', 'yhat_upper']
    forecast[cols_to_fix] = forecast[cols_to_fix].clip(lower=0)
    
    return m, forecast

# --- 儀表板繪圖函數 (新功能) ---
def plot_gauge(current_price, future_price):
    # 計算潛在漲跌幅
    change_pct = ((future_price - current_price) / current_price) * 100
    
    # 決定建議文字與顏色
    if change_pct >= 10:
        rating = "強烈買進 (Strong Buy)"
        color = "#00CC96" # 鮮綠
    elif change_pct >= 5:
        rating = "買進 (Buy)"
        color = "#2ca02c" # 綠
    elif change_pct > -5:
        rating = "持守 (Hold)"
        color = "#ffbf00" # 黃
    elif change_pct > -10:
        rating = "賣出 (Sell)"
        color = "#d62728" # 紅
    else:
        rating = "強烈賣出 (Strong Sell)"
        color = "#8c1515" # 深紅

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = change_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"AI 建議: {rating}", 'font': {'size': 20}},
        delta = {'reference': 0, 'position': "top", 'valueformat': ".2f", 'suffix': "%"},
        number = {'suffix': "%", 'font': {'color': color}},
        gauge = {
            'axis': {'range': [-30, 30], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white", 'thickness': 0.2}, # 指針顏色
            'bgcolor': "black",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-30, -10], 'color': '#8c1515'}, # 深紅
                {'range': [-10, -5], 'color': '#d62728'}, # 紅
                {'range': [-5, 5], 'color': '#ffbf00'},    # 黃 (持守區)
                {'range': [5, 10], 'color': '#2ca02c'},    # 綠
                {'range': [10, 30], 'color': '#00CC96'}    # 鮮綠
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': change_pct
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#0E1117", # 配合 Streamlit 深色背景
        font={'color': "white"}
    )
    return fig

# --- 主程式邏輯 ---
if ticker_input:
    ticker_symbol = ticker_input.upper().strip()
    
    with st.spinner(f'正在分析 {ticker_symbol} 的投資光譜...'):
        hist = get_stock_data(ticker_symbol)

        if hist is None or hist.empty:
            st.error(f"❌ 找不到代碼 '{ticker_symbol}'。")
        else:
            # 1. 基礎數據
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            delta = current_price - prev_price
            delta_percent = (delta / prev_price) * 100
            color_code = "green" if delta >= 0 else "red"
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 20px;">
                <h3 style="margin:0; color: #aaa;">{ticker_symbol} 現價</h3>
                <h1 style="margin:0; color: {color_code};">${current_price:.2f}</h1>
                <p style="margin:0; color: {color_code};">{delta:+.2f} ({delta_percent:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            try:
                m, forecast = predict_stock(hist, forecast_days)
                
                # 2. 【新功能】建議光譜儀表板
                # 取得預測期最後一天的價格
                future_price = forecast['yhat'].iloc[-1]
                st.subheader("🧭 AI 建議光譜 (Recommendation Spectrum)")
                gauge_chart = plot_gauge(current_price, future_price
