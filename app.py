import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 股市預測 v8.0", layout="wide")
st.title("🤖 AI 股市預測 v8.0")
st.caption("雙市場版：支援 🇺🇸 美股 & 🇹🇼 台股 (自動偵測上市櫃)")

# --- 2. 側邊欄：市場選擇 ---
st.sidebar.header("設定")
market_mode = st.sidebar.radio("選擇市場", ["🇺🇸 美股 (US)", "🇹🇼 台股 (TW)"])

# --- 3. 輸入區 (根據市場變換預設值) ---
col_input, col_days = st.columns([2, 1])

with col_input:
    if market_mode == "🇺🇸 美股 (US)":
        default_ticker = "NVDA"
        label_text = "請輸入美股代碼 (如 NVDA, TSLA)"
        currency = "USD"
        currency_symbol = "$"
    else:
        default_ticker = "2330"
        label_text = "請輸入台股代碼 (如 2330, 0050)"
        currency = "TWD"
        currency_symbol = "NT$"
        
    ticker_input = st.text_input(label_text, value=default_ticker)

with col_days:
    forecast_days = st.selectbox("預測範圍", [30, 60, 90, 180], index=1)

# --- 4. 資料獲取函數 (含台股自動後綴偵測) ---
@st.cache_data
def get_stock_data(ticker, market):
    try:
        # 台股邏輯：使用者通常只打數字，需自動測試 .TW 或 .TWO
        if market == "🇹🇼 台股 (TW)":
            # 如果使用者沒打後綴，我們先試 .TW (上市)
            if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
                test_ticker = f"{ticker}.TW"
            else:
                test_ticker = ticker
            
            stock = yf.Ticker(test_ticker)
            hist = stock.history(period="5y", auto_adjust=True)
            
            # 如果 .TW 抓不到，改試 .TWO (上櫃)
            if hist is None or hist.empty:
                test_ticker = f"{ticker}.TWO"
                stock = yf.Ticker(test_ticker)
                hist = stock.history(period="5y", auto_adjust=True)
        else:
            # 美股邏輯
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y", auto_adjust=True)

        # 備用：如果 auto_adjust 失敗，抓原始資料
        if hist is None or hist.empty:
            hist = stock.history(period="5y", auto_adjust=False)
        
        if hist is None or hist.empty:
            return None, None, None # 回傳 ticker 用於顯示最終抓到的代碼

        hist.reset_index(inplace=True)
        if 'Date' in hist.columns:
             hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        info = stock.info
        # 回傳抓到的正確代碼 (例如 2330 -> 2330.TW)
        real_symbol = stock.ticker 
        return hist, info, real_symbol

    except Exception:
        return None, None, None

# --- 5. AI 預測函數 ---
def predict_stock(data, days):
    df_train = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet(daily_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=days, freq='B')
    forecast = m.predict(future)
    
    cols_to_fix = ['yhat', 'yhat_lower', 'yhat_upper']
    forecast[cols_to_fix] = forecast[cols_to_fix].clip(lower=0)
    return m, forecast

# --- 6. 回測函數 ---
def backtest_model(data, test_days=5):
    df_full = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    train_df = df_full.iloc[:-test_days]
    test_df = df_full.iloc[-test_days:].copy()
    
    m = Prophet(daily_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=test_days, freq='B')
    forecast = m.predict(future)
    
    forecast_tail = forecast.tail(test_days)[['ds', 'yhat']]
    result = pd.merge(test_df, forecast_tail, on='ds', how='inner')
    result['error_pct'] = ((result['y'] - result['yhat']).abs() / result['y']) * 100
    acc_score = 100 - result['error_pct'].mean()
    return acc_score, result

# --- 7. 繪圖輔助函數 ---
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
                {'range': [-5, 5], 'color': '#ffbf00'}, {'range': [5, 10], 'color': '#2ca02c'},
                {'range': [10, 30], 'color': '#00CC96'}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': change_pct}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor="#0E1117", font={'color': "white"})
    return fig, change_pct

def get_ai_explanation(ticker, days, pct):
    if pct >= 10: return f"🚀 **強烈看漲**：{ticker} 動能強勁 (>10%)，多頭排列穩固。"
    elif pct >= 5: return f"📈 **看漲**：{ticker} 呈溫和上升趨勢，適合佈局。"
    elif pct > -5: return f"⚖️ **持守**：{ticker} 預期區間震盪，建議觀望。"
    elif pct > -10: return f"📉 **看跌**：動能轉弱，{ticker} 面臨回調壓力。"
    else: return f"⚠️ **強烈看跌**：{ticker} 下行風險高，建議避開。"

def format_large_number(num, c_symbol):
    if num is None: return "N/A"
    # 台股習慣看「億」，美股看「B/T」
    if c_symbol == "NT$":
        return f"{num/1e8:.1f}億"
    else:
        if num >= 1e12: return f"{num/1e12:.2f}T"
        if num >= 1e9: return f"{num/1e9:.2f}B"
        return f"{num/1e6:.2f}M"

# --- 8. 主程式執行區 ---
if ticker_input:
    ticker_clean = ticker_input.upper().strip()
    
    with st.spinner(f'AI 正在搜尋 {market_mode} 數據...'):
        # 接收三個值：歷史數據, 基本面, 真實代碼(含後綴)
        hist, info, real_symbol = get_stock_data(ticker_clean, market_mode)

        if hist is None or hist.empty:
            st.error(f"❌ 找不到代碼 '{ticker_clean}'")
            if market_mode == "🇹🇼 台股 (TW)":
                st.info("💡 提示：台股請輸入數字代碼，如 2330 (台積電), 2603 (長榮)。")
        else:
            # (A) 顯示價格
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            delta = current_price - prev_price
            pct = (delta / prev_price) * 100
            color = "green" if delta >= 0 else "red"
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #262730; margin-bottom: 20px;">
                <h3 style="margin:0; color: #aaa;">{real_symbol} ({currency})</h3>
                <h1 style="margin:0; color: {color};">{currency_symbol}{current_price:.2f}</h1>
                <p style="margin:0; color: {color};">{delta:+.2f} ({pct:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            # (B) 基本面
            if info:
                st.subheader("📊 基本面健檢")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("市值", format_large_number(info.get('marketCap'), currency_symbol))
                with c2: st.metric("PE (本益比)", f"{info.get('trailingPE', 'N/A')}")
                with c3: st.metric("EPS", f"{info.get('trailingEps', 'N/A')}")
                with c4: st.metric("52週高", f"{currency_symbol}{info.get('fiftyTwoWeekHigh', 0)}")
                st.divider()

            try:
                # (C) AI 預測 & 儀表板
                m, forecast = predict_stock(hist, forecast_days)
                future_price = forecast['yhat'].iloc[-1]

                st.subheader("🧭 AI 建議光譜")
                gauge, chg_pct = plot_gauge(current_price, future_price, currency_symbol)
                st.plotly_chart(gauge, use_container_width=True)
                st.info(get_ai_explanation(real_symbol, forecast_days, chg_pct))

                # (D) 走勢圖
                st.subheader("📈 詳細走勢預測")
                fig = plot_plotly(m, forecast)
                fig.update_layout(xaxis_title=None, yaxis_title=currency, hovermode="x", height=500, margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)

                # (E) 準確度回測 (Backtest)
                st.divider()
                st.subheader("🕵️‍♂️ 模型真實準確度回測")
                with st.expander(f"查看 {real_symbol} 近期預測準確度", expanded=True):
                    acc, bt_df = backtest_model(hist)
                    score_color = "green" if acc >= 90 else "orange" if acc >= 80 else "red"
                    st.markdown(f"<h3 style='text-align:center'>近期評分: <span style='color:{score_color}'>{acc:.1f} 分</span></h3>", unsafe_allow_html=True)
                    
                    bt_display = bt_df[['ds', 'y', 'yhat', 'error_pct']].copy()
                    bt_display.columns = ['日期', '真實價', '預測價', '誤差%']
                    bt_display['日期'] = bt_display['日期'].dt.strftime('%m-%d')
                    st.dataframe(bt_display.style.format({'真實價': '{:.2f}', '預測價': '{:.2f}', '誤差%': '{:.2f}%'}), use_container_width=True)

            except Exception as e:
                st.error(f"分析失敗: {e}")
