import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np
from datetime import datetime

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 股市戰情室 v12.1", layout="wide")
st.title("🤖 AI 股市戰情室 v12.1")
st.caption("修復版：解決資訊卡顯示為原始碼的問題")

# --- 2. 輸入與設定區 ---
st.markdown("### 1️⃣ 選擇市場")
market_mode = st.radio(
    "選擇市場", 
    ["🇺🇸 美股 (US)", "🇹🇼 台股 (TW)"], 
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("### 2️⃣ 輸入代碼")
col_input, col_days = st.columns([2, 1])

with col_input:
    if market_mode == "🇺🇸 美股 (US)":
        default_ticker = "NVDA"
        label_text = "美股代碼 (如 NVDA, TSLA)"
        currency = "USD"
        currency_symbol = "$"
    else:
        default_ticker = "2330"
        label_text = "台股代碼 (如 2330, 2603)"
        currency = "TWD"
        currency_symbol = "NT$"
        
    ticker_input = st.text_input(label_text, value=default_ticker)

with col_days:
    forecast_days = st.selectbox("預測天數", [30, 60, 90, 180], index=1)

# --- 3. 資料獲取函數 ---
@st.cache_data
def get_stock_data(ticker, market):
    try:
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
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y", auto_adjust=True)

        if hist is None or hist.empty:
            hist = stock.history(period="5y", auto_adjust=False)
        
        if hist is None or hist.empty:
            return None, None, None, None

        hist.reset_index(inplace=True)
        if 'Date' in hist.columns:
             hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        try:
            intraday = stock.history(period="1d", interval="5m", auto_adjust=True)
            if intraday is not None and not intraday.empty:
                intraday.reset_index(inplace=True)
                if 'Datetime' in intraday.columns:
                    intraday['Datetime'] = intraday['Datetime'].dt.tz_localize(None)
            else:
                intraday = None
        except:
            intraday = None
        
        info = stock.info
        real_symbol = stock.ticker 
        return hist, info, real_symbol, intraday

    except Exception:
        return None, None, None, None

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

# --- 6. 繪圖與格式化函數 ---
def plot_gauge(current, future, c_symbol):
    raw_change_pct = ((future - current) / current) * 100
    change_pct = round(raw_change_pct, 3)
    
    if change_pct >= 10: rating, color = "強烈買進", "#00CC96"
    elif change_pct >= 5: rating, color = "買進", "#2ca02c"
    elif change_pct > -5: rating, color = "持守", "#ffbf00"
    elif change_pct > -10: rating, color = "賣出", "#d62728"
    else: rating, color = "強烈賣出", "#8c1515"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = change_pct,
        title = {'text': f"AI 建議: {rating}", 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'color': color}, 'valueformat': "+.3f"},
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

def plot_intraday(intraday_data, symbol, currency_symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=intraday_data['Datetime'],
        open=intraday_data['Open'], high=intraday_data['High'],
        low=intraday_data['Low'], close=intraday_data['Close'],
        name="Price"
    ))
    fig.update_layout(
        title=dict(text=f"📉 當日走勢 (5分K)", font=dict(size=14, color="#ccc")),
        xaxis_rangeslider_visible=False, height=300,
        margin=dict(l=10, r=10, t=40, b=20),
        paper_bgcolor="#1e212b", plot_bgcolor="#1e212b",
        font=dict(color="#aaa"),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#333", title=currency_symbol)
    )
    return fig

def get_ai_explanation(ticker, days, pct):
    if pct >= 10: return f"🚀 **強烈看漲**：{ticker} 動能強勁 (>10%)，多頭排列穩固。"
    elif pct >= 5: return f"📈 **看漲**：{ticker} 呈溫和上升趨勢，適合佈局。"
    elif pct > -5: return f"⚖️ **持守**：{ticker} 預期區間震盪，建議觀望。"
    elif pct > -10: return f"📉 **看跌**：動能轉弱，{ticker} 面臨回調壓力。"
    else: return f"⚠️ **強烈看跌**：{ticker} 下行風險高，建議避開。"

def format_large_number(num, c_symbol):
    if num is None: return "N/A"
    if c_symbol == "NT$":
        if num >= 1e12: return f"{num/1e12:.2f}兆"
        return f"{num/1e8:.2f}億"
    else:
        if num >= 1e12: return f"{num/1e12:.2f}T"
        if num >= 1e9: return f"{num/1e9:.2f}B"
        return f"{num/1e6:.2f}M"

# --- 8. 主程式執行區 ---
if ticker_input:
    ticker_clean = ticker_input.upper().strip()
    
    with st.spinner(f'AI 正在搜尋 {market_mode} 數據...'):
        hist, info, real_symbol, intraday = get_stock_data(ticker_clean, market_mode)

        if hist is None or hist.empty:
            st.error(f"❌ 找不到代碼 '{ticker_clean}'")
            if market_mode == "🇹🇼 台股 (TW)":
                st.info("💡 提示：台股請輸入數字代碼，如 2330 (台積電), 2603 (長榮)。")
        else:
            # (A) 全能資訊卡 (Unified Info Card)
            last_row = hist.iloc[-1]
            current_price = last_row['Close']
            prev_price = hist.iloc[-2]['Close']
            delta = current_price - prev_price
            pct = (delta / prev_price) * 100
            color = "#00CC96" if delta >= 0 else "#FF4B4B"
            
            day_open = last_row['Open']
            day_high = last_row['High']
            day_low = last_row['Low']
            day_vol = format_large_number(last_row['Volume'], currency_symbol)
            
            mkt_cap = format_large_number(info.get('marketCap'), currency_symbol)
            pe_ratio = f"{info.get('trailingPE', 'N/A')}"
            eps = f"{info.get('trailingEps', 'N/A')}"
            high_52 = f"{info.get('fiftyTwoWeekHigh', 'N/A')}"

            # 【修復重點】移除 HTML 字串的所有前方縮排，避免被判定為程式碼區塊
            card_html = f"""
<div style="background-color: #1e212b; border-radius: 15px; padding: 20px; border: 1px solid #444; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 15px;">
        <div>
            <h3 style="margin:0; color: #ccc; font-size: 1.2em;">{real_symbol}</h3>
            <div style="display: flex; align-items: baseline; gap: 10px;">
                <h1 style="margin:0; font-size: 2.8em; color: {color};">{currency_symbol}{current_price:.2f}</h1>
                <span style="font-size: 1.2em; color: {color}; font-weight: bold;">{delta:+.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding-bottom: 15px;">
        <div style="text-align: center;"><div style="color: #888; font-size: 0.8em;">開盤</div><div style="font-weight: bold; color: #eee;">{day_open:.2f}</div></div>
        <div style="text-align: center;"><div style="color: #888; font-size: 0.8em;">最高</div><div style="font-weight: bold; color: #eee;">{day_high:.2f}</div></div>
        <div style="text-align: center;"><div style="color: #888; font-size: 0.8em;">最低</div><div style="font-weight: bold; color: #eee;">{day_low:.2f}</div></div>
        <div style="text-align: center;"><div style="color: #888; font-size: 0.8em;">量</div><div style="font-weight: bold; color: #eee;">{day_vol}</div></div>
    </div>
    <div style="border-top: 1px dashed #444; margin: 0 0 15px 0;"></div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
        <div style="text-align: center;"><div style="color: #aaa; font-size: 0.8em;">市值</div><div style="color: #ddd;">{mkt_cap}</div></div>
        <div style="text-align: center;"><div style="color: #aaa; font-size: 0.8em;">本益比</div><div style="color: #ddd;">{pe_ratio}</div></div>
        <div style="text-align: center;"><div style="color: #aaa; font-size: 0.8em;">EPS</div><div style="color: #ddd;">{eps}</div></div>
        <div style="text-align: center;"><div style="color: #aaa; font-size: 0.8em;">52週高</div><div style="color: #ddd;">{high_52}</div></div>
    </div>
</div>
"""
            st.markdown(card_html, unsafe_allow_html=True)

            # (B) 走勢圖
            if intraday is not None and not intraday.empty:
                intraday_chart = plot_intraday(intraday, real_symbol, currency_symbol)
                st.plotly_chart(intraday_chart, use_container_width=True)
            else:
                st.caption("💤 目前無即時分時數據")

            st.divider()

            try:
                # (C) AI 預測
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

                # (E) 回測
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
