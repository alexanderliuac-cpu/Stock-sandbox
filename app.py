import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 美股預測 v5.3", layout="wide")
st.title("🤖 AI 美股預測 v5.3")
st.caption("儀表板單一數值化 & 新增 AI 判讀解釋")

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
            return None

        hist.reset_index(inplace=True)
        if 'Date' in hist.columns:
             hist['Date'] = hist['Date'].dt.tz_localize(None)
        
        return hist
    except Exception:
        return None

# --- 4. AI 預測函數 ---
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

# --- 5. 儀表板與解釋函數 (更新重點) ---
def plot_gauge(current_price, future_price):
    change_pct = ((future_price - current_price) / current_price) * 100
    
    if change_pct >= 10:
        rating, color = "強烈買進", "#00CC96"
    elif change_pct >= 5:
        rating, color = "買進", "#2ca02c"
    elif change_pct > -5:
        rating, color = "持守", "#ffbf00"
    elif change_pct > -10:
        rating, color = "賣出", "#d62728"
    else:
        rating, color = "強烈賣出", "#8c1515"

    # 【優化】mode 改為 "gauge+number"，移除重複的 delta
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = change_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"AI 建議: {rating}", 'font': {'size': 20}},
        # 設定數值格式，強制顯示正負號 (+.1f)
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
        paper_bgcolor="#0E1117", 
        font={'color': "white"}
    )
    return fig, change_pct

def get_ai_explanation(ticker, days, change_pct):
    """生成 AI 判讀文字"""
    if change_pct >= 10:
        return f"🚀 **強烈看漲理由**：模型預測 {ticker} 在未來 {days} 天擁有強勁上漲動能 (>10%)。歷史趨勢顯示多頭排列穩固，建議積極佈局。"
    elif change_pct >= 5:
        return f"📈 **看漲理由**：預測 {ticker} 呈現溫和上升趨勢。雖然不如爆發性增長，但預期回報為正 ({change_pct:.1f}%)，適合分批買入。"
    elif change_pct > -5:
        return f"⚖️ **持守理由**：模型顯示 {ticker} 短期內缺乏明確方向，預期將在區間內震盪。風險與報酬相對平衡，建議暫時觀望。"
    elif change_pct > -10:
        return f"📉 **看跌理由**：動能轉弱，預測未來 {days} 天面臨回調壓力。除非有重大利多，否則價格可能緩步走低，建議減碼。"
    else:
        return f"⚠️ **強烈看跌理由**：模型偵測到顯著的下行風險，預期跌幅可能超過 10%。技術面呈現疲軟，建議避開或設定嚴格停損。"

# --- 6. 主程式執行區 ---
if ticker_input:
    ticker_symbol = ticker_input.upper().strip()
    
    with st.spinner(f'正在分析 {ticker_symbol} 的投資光譜...'):
        hist = get_stock_data(ticker_symbol)

        if hist is None or hist.empty:
            st.error(f"❌ 找不到代碼 '{ticker_symbol}'")
        else:
            # (A) 顯示目前價格
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

            try:
                # AI 預測
                m, forecast = predict_stock(hist, forecast_days)
                future_price = forecast['yhat'].iloc[-1]

                # (B) 顯示儀表板 (單一數值)
                st.subheader("🧭 AI 建議光譜")
                gauge_chart, change_pct = plot_gauge(current_price, future_price)
                st.plotly_chart(gauge_chart, use_container_width=True)
                
                # (C) 【新功能】AI 判讀文字
                explanation = get_ai_explanation(ticker_symbol, forecast_days, change_pct)
                st.info(explanation)

                # (D) 走勢圖
                st.subheader("📈 詳細走勢預測")
                fig = plot_plotly(m, forecast)
                fig.update_layout(
                    xaxis_title=None, yaxis_title="USD", 
                    hovermode="x", height=500,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # (E) 未來 10 天數據
                st.subheader("📅 未來 10 天預測表")
                last_date = hist['Date'].iloc[-1]
                future_only = forecast[forecast['ds'] > last_date]
                future_data = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10)
                
                future_data.columns = ['日期', '預測價', '下限', '上限']
                future_data['日期'] = future_data['日期'].dt.strftime('%m-%d (%a)')
                
                st.dataframe(
                    future_data.style.format("{:.2f}", subset=['預測價', '下限', '上限']),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"分析失敗: {e}")
