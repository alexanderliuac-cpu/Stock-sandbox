import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 美股預測 v7.0", layout="wide")
st.title("🤖 AI 美股預測 v7.0")
st.caption("旗艦版：基本面 + 趨勢預測 + 準確度回測 (Backtest)")

# --- 2. 輸入區 ---
col_input, col_days = st.columns([2, 1])

with col_input:
    ticker_input = st.text_input("請輸入美股代碼", value="NVDA")

with col_days:
    forecast_days = st.selectbox("預測範圍", [30, 60, 90, 180], index=1)

# --- 3. 資料獲取函數 ---
@st.cache_data
def get_stock_data(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="5y", auto_adjust=True)
        if hist is None or hist.empty:
            hist = stock.history(period="5y", auto_adjust=False)
        
        if hist is None or hist.empty:
            return None, None

        hist.reset_index(inplace=True)
        if 'Date' in hist.columns:
             hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        info = stock.info
        return hist, info
    except Exception:
        return None, None

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

# --- 5. 【新功能】回測函數 ---
def backtest_model(data, test_days=5):
    """
    時光機回測：隱藏最後 N 天的數據，用剩下的數據訓練，
    然後預測這 N 天，比較「預測值」與「真實值」的誤差。
    """
    # 準備數據
    df_full = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # 切割數據：訓練集 (扣除最後 N 天)
    train_df = df_full.iloc[:-test_days]
    # 測試集 (真實的最後 N 天)
    test_df = df_full.iloc[-test_days:].copy()
    
    # 訓練模型 (用過去的數據)
    m = Prophet(daily_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(train_df)
    
    # 預測這 N 天
    future = m.make_future_dataframe(periods=test_days, freq='B')
    forecast = m.predict(future)
    
    # 提取預測結果中對應的日期
    forecast_tail = forecast.tail(test_days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # 合併 真實數據 與 預測數據
    # 注意：有些日期可能是假日，merge 會自動對齊
    result = pd.merge(test_df, forecast_tail, on='ds', how='inner')
    
    # 計算誤差
    result['error'] = result['y'] - result['yhat']
    result['error_pct'] = (result['error'].abs() / result['y']) * 100
    
    # 計算平均誤差 (MAPE)
    mape = result['error_pct'].mean()
    accuracy_score = 100 - mape
    
    return accuracy_score, result

# --- 6. 繪圖與輔助函數 ---
def plot_gauge(current_price, future_price):
    change_pct = ((future_price - current_price) / current_price) * 100
    
    if change_pct >= 10: rating, color = "強烈買進", "#00CC96"
    elif change_pct >= 5: rating, color = "買進", "#2ca02c"
    elif change_pct > -5: rating, color = "持守", "#ffbf00"
    elif change_pct > -10: rating, color = "賣出", "#d62728"
    else: rating, color = "強烈賣出", "#8c1515"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = change_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"AI 建議: {rating}", 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'color': color}, 'valueformat': "+.1f"},
        gauge = {
            'axis': {'range': [-30, 30], 'tickwidth': 1},
            'bar': {'color': "white", 'thickness': 0.2},
            'bgcolor': "black",
            'steps': [
                {'range': [-30, -10], 'color': '#8c1515'},
                {'range': [-10, -5], 'color': '#d62728'},
                {'range': [-5, 5], 'color': '#ffbf00'},
                {'range': [5, 10], 'color': '#2ca02c'},
                {'range': [10, 30], 'color': '#00CC96'}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': change_pct}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="#0E1117", font={'color': "white"})
    return fig, change_pct

def get_ai_explanation(ticker, days, change_pct):
    if change_pct >= 10: return f"🚀 **強烈看漲**：預測 {ticker} 動能強勁 (>10%)。"
    elif change_pct >= 5: return f"📈 **看漲**：預測 {ticker} 呈溫和上升趨勢。"
    elif change_pct > -5: return f"⚖️ **持守**：缺乏方向，預期區間震盪。"
    elif change_pct > -10: return f"📉 **看跌**：動能轉弱，建議減碼。"
    else: return f"⚠️ **強烈看跌**：下行風險高，建議避開。"

def format_large_number(num):
    if num is None: return "N/A"
    if num >= 1e12: return f"{num/1e12:.2f}T"
    if num >= 1e9: return f"{num/1e9:.2f}B"
    if num >= 1e6: return f"{num/1e6:.2f}M"
    return f"{num:.2f}"

# --- 7. 主程式執行區 ---
if ticker_input:
    ticker_symbol = ticker_input.upper().strip()
    
    with st.spinner(f'AI 正在進行深度分析與回測驗證...'):
        hist, info = get_stock_data(ticker_symbol)

        if hist is None or hist.empty:
            st.error(f"❌ 找不到代碼 '{ticker_symbol}'")
        else:
            # (A) 基本資訊
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            delta = current_price - prev_price
            delta_pct = (delta / prev_price) * 100
            color_code = "green" if delta >= 0 else "red"
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 20px;">
                <h3 style="margin:0; color: #aaa;">{ticker_symbol} 現價</h3>
                <h1 style="margin:0; color: {color_code};">${current_price:.2f}</h1>
                <p style="margin:0; color: {color_code};">{delta:+.2f} ({delta_pct:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            # (B) 基本面卡片
            if info:
                st.subheader("📊 基本面健檢")
                f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                with f_col1: st.metric("市值", format_large_number(info.get('marketCap')))
                with f_col2: st.metric("PE", f"{info.get('trailingPE', 0):.2f}")
                with f_col3: st.metric("EPS", f"{info.get('trailingEps', 0):.2f}")
                with f_col4: st.metric("52週高", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                st.divider()

            try:
                # (C) 未來預測
                m, forecast = predict_stock(hist, forecast_days)
                future_price = forecast['yhat'].iloc[-1]

                st.subheader("🧭 AI 建議光譜")
                gauge_chart, change_pct = plot_gauge(current_price, future_price)
                st.plotly_chart(gauge_chart, use_container_width=True)
                st.info(get_ai_explanation(ticker_symbol, forecast_days, change_pct))

                st.subheader("📈 詳細走勢預測")
                fig = plot_plotly(m, forecast)
                fig.update_layout(xaxis_title=None, yaxis_title="USD", hovermode="x", height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📅 未來 10 天預測表")
                last_date = hist['Date'].iloc[-1]
                future_only = forecast[forecast['ds'] > last_date]
                future_data = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10)
                future_data.columns = ['日期', '預測價', '下限', '上限']
                future_data['日期'] = future_data['日期'].dt.strftime('%m-%d (%a)')
                st.dataframe(future_data.style.format("{:.2f}", subset=['預測價', '下限', '上限']), use_container_width=True)

                # (D) 【新功能】準確度回測區塊
                st.divider()
                st.subheader("🕵️‍♂️ 模型真實準確度回測 (Backtest)")
                
                with st.expander("點擊查看：這模型過去 5 天準嗎？", expanded=True):
                    # 執行回測 (隱藏最後 5 天來測試)
                    acc_score, backtest_df = backtest_model(hist, test_days=5)
                    
                    # 顯示分數
                    score_color = "green" if acc_score >= 90 else "orange" if acc_score >= 80 else "red"
                    st.markdown(f"""
                    <h3 style="text-align:center;">近期準確度評分: <span style="color:{score_color}">{acc_score:.1f} 分</span></h3>
                    <p style="text-align:center; color:#888;">(滿分 100，分數越高代表近期預測越貼近真實走勢)</p>
                    """, unsafe_allow_html=True)
                    
                    # 整理表格
                    display_df = backtest_df[['ds', 'y', 'yhat', 'error_pct']].copy()
                    display_df.columns = ['日期', '真實收盤價', 'AI 預測價', '誤差 %']
                    display_df['日期'] = display_df['日期'].dt.strftime('%m-%d')
                    
                    st.dataframe(
                        display_df.style.format({
                            "真實收盤價": "{:.2f}", 
                            "AI 預測價": "{:.2f}", 
                            "誤差 %": "{:.2f}%"
                        }),
                        use_container_width=True
                    )
                    st.caption("原理：我們將過去 5 天的數據隱藏起來，讓 AI 重新預測一次，並與真實發生的價格對答案。")

            except Exception as e:
                st.error(f"分析失敗: {e}")
