import streamlit as st
import pandas as pd

# --- 設定頁面 ---
st.set_page_config(page_title="航空座椅比一比", layout="wide", page_icon="✈️")

# --- CSS 樣式優化 (讓 metric 更好看) ---
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 24px; }
    .stProgress > div > div > div > div { background-color: #005f73; }
</style>
""", unsafe_allow_html=True)

# --- 資料庫 ---
# 這裡是用 Python 字典儲存數據，比 JS 更易讀
DATA = {
    "長榮航空 (EVA Air)": [
        {"name": "777-300ER 皇璽桂冠艙", "pitch": 78, "width": 26, "amenities": ["全平躺", "16吋螢幕", "睡衣", "防噪耳機"]},
        {"name": "777-300ER 豪華經濟艙", "pitch": 38, "width": 19.5, "amenities": ["11吋螢幕", "USB充電", "專屬過夜包"]},
        {"name": "777-300ER 經濟艙", "pitch": 32, "width": 18.3, "amenities": ["11吋螢幕", "USB充電"]}
    ],
    "中華航空 (China Airlines)": [
        {"name": "A350-900 豪華商務艙", "pitch": 78, "width": 28, "amenities": ["全平躺", "18吋螢幕", "Sky Lounge", "防噪耳機"]},
        {"name": "A350-900 豪華經濟艙", "pitch": 39, "width": 20, "amenities": ["固定式椅背", "12吋螢幕", "專屬閱讀燈"]},
        {"name": "A350-900 經濟艙", "pitch": 32, "width": 18, "amenities": ["親子臥艙(選配)", "11吋螢幕"]}
    ],
    "星宇航空 (Starlux)": [
        {"name": "A350-900 頭等艙", "pitch": 83, "width": 32, "amenities": ["全平躺", "4K 32吋螢幕", "拉門隱私", "零重力模式"]},
        {"name": "A350-900 商務艙", "pitch": 80, "width": 28, "amenities": ["全平躺", "4K 24吋螢幕", "拉門隱私", "無線充電"]},
        {"name": "A350-900 經濟艙", "pitch": 31, "width": 18.3, "amenities": ["4K 13吋螢幕", "藍牙音訊"]}
    ],
    "全日空 (ANA)": [
        {"name": "777-300ER The Room (商務)", "pitch": 64, "width": 38, "amenities": ["全平躺", "超寬座椅", "4K 24吋螢幕", "拉門隱私"]},
        {"name": "787-9 經濟艙", "pitch": 34, "width": 17.3, "amenities": ["業界領先椅距", "9吋螢幕", "腳踏板"]}
    ],
    "阿聯酋 (Emirates)": [
        {"name": "A380 頭等艙", "pitch": 86, "width": 23, "amenities": ["全平躺", "機上淋浴間", "私人套房", "32吋螢幕"]},
        {"name": "A380 經濟艙", "pitch": 32, "width": 18, "amenities": ["13.3吋螢幕", "ICE娛樂系統"]}
    ]
}

st.title("✈️ 航空公司座椅終極比一比")
st.markdown("選擇三個選手，比較他們的椅距 (Pitch)、椅寬 (Width) 與設備。")
st.divider()

# --- 建立三個比較欄位 ---
cols = st.columns(3)

selected_seats = []

# 使用迴圈建立三個同樣的控制項
for i, col in enumerate(cols):
    with col:
        st.subheader(f"選手 {i+1}")
        
        # 1. 選擇航空公司
        airline = st.selectbox(
            f"選擇航空公司 ({i+1})", 
            options=["請選擇"] + list(DATA.keys()), 
            key=f"airline_{i}"
        )
        
        # 2. 選擇機型/艙等
        if airline != "請選擇":
            seat_options = [s['name'] for s in DATA[airline]]
            seat_name = st.selectbox(
                f"選擇艙等 ({i+1})", 
                options=seat_options,
                key=f"seat_{i}"
            )
            
            # 找出選到的那個座位資料
            seat_data = next(s for s in DATA[airline] if s['name'] == seat_name)
            selected_seats.append(seat_data)
            
            st.markdown("---")
            
            # 3. 顯示數據
            # 椅距
            st.metric("椅距 (Pitch)", f"{seat_data['pitch']} 吋")
            # 視覺化進度條 (假設最大90吋)
            st.progress(min(seat_data['pitch'] / 90, 1.0))
            
            # 椅寬
            st.metric("椅寬 (Width)", f"{seat_data['width']} 吋")
            # 視覺化進度條 (假設最大40吋)
            st.progress(min(seat_data['width'] / 40, 1.0))
            
            # 設備標籤
            st.write("**特色設備:**")
            for item in seat_data['amenities']:
                if "平躺" in item or "4K" in item or "拉門" in item:
                    st.success(item) # 綠色高亮
                else:
                    st.info(item) # 藍色普通
        else:
            st.info("請先選擇航空公司")
            selected_seats.append(None)

# --- 底部總結比較 (選擇性) ---
if any(selected_seats):
    st.divider()
    st.subheader("📊 數據直接並排")
    
    # 整理成 DataFrame 做表格比較
    comp_data = []
    for idx, seat in enumerate(selected_seats):
        if seat:
            comp_data.append({
                "選手": f"選手 {idx+1}",
                "艙等": seat['name'],
                "椅距 (吋)": seat['pitch'],
                "椅寬 (吋)": seat['width']
            })
    
    if comp_data:
        df = pd.DataFrame(comp_data)
        st.dataframe(df.set_index("選手"), use_container_width=True)
        
        # 使用 Streamlit 內建圖表
        st.caption("椅距對比圖")
        st.bar_chart(df.set_index("艙等")["椅距 (吋)"])
