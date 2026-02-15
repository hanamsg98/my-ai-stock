import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
import google.generativeai as genai
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# --- 1. 화면 기본 설정 및 디자인 ---
st.set_page_config(page_title="Pro AI Trader", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; font-family: 'Roboto', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    div[data-testid="metric-container"] { background-color: #21262D; border: 1px solid #30363D; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s; }
    div[data-testid="metric-container"]:hover { transform: translateY(-2px); }
    .stTabs [data-baseweb="tab"] { background-color: #161B22; border-radius: 5px 5px 0 0; color: #8B949E; font-weight: 600; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    div.stButton > button { background-color: #238636; color: white; border: none; font-weight: bold; }
    div.stButton > button:hover { background-color: #2EA043; }
    .big-font { font-size: 1.5rem !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- [NEW] 1-2. 왼쪽 사이드바: 리스크 관리 알리미 ---
with st.sidebar:
    st.markdown("## 🚨 Risk & Macro Calendar")
    st.info("단기 매매 전, 변동성 폭발 일정을 반드시 확인하세요.")
    
    # 2~3월 주요 일정 (실제 일정에 맞게 수정 가능)
    events = {
        "날짜": ["02/18 (수)", "02/26 (목)", "03/06 (금)", "03/11 (수)", "03/18 (수)"],
        "시간": ["04:00", "22:30", "22:30", "22:30", "03:00"],
        "이벤트": ["FOMC 의사록 공개", "미국 PCE물가지수", "미국 고용보고서", "미국 CPI 발표", "FOMC 금리 결정"],
        "중요도": ["⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
    }
    df_events = pd.DataFrame(events)
    st.dataframe(df_events, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ✅ 스캘핑/종배 Check List")
    st.checkbox("🇺🇸 미국장 나스닥/반도체 등락 확인했는가?")
    st.checkbox("🚨 VIX(공포지수)가 안정권인가?")
    st.checkbox("🔥 국내 주도주 섹터로 수급이 몰리는가?")
    st.checkbox("🌙 오늘 밤 중요 매크로 발표가 없는가?")
    
    st.markdown("---")
    st.caption("위 4가지 체크리스트에 모두 체크하지 못했다면, 보수적으로 비중을 줄이거나 관망하세요.")

st.title("🚀 Pro AI Trading Dashboard")
st.markdown("---")

# --- 2. 데이터 수집 함수 ---
@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y"):
    try:
        if ticker.startswith("^") or ticker.endswith("=X") or ticker.endswith("=F") or ticker.endswith("NYB"):
             df = yf.Ticker(ticker).history(period=period)
        elif ticker.isdigit():
             end_date = datetime.date.today()
             start_date = end_date - datetime.timedelta(days=730)
             df = fdr.DataReader(ticker, start_date, end_date)
             df = df.rename(columns={'Close': 'Close', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume'})
             df.index.name = 'Date'
             days_to_keep = 30 if 'mo' in period else (365 if period == "1y" else 730)
             df = df.loc[df.index >= (pd.Timestamp.now() - pd.Timedelta(days=days_to_keep))]
        else:
             df = yf.Ticker(ticker).history(period=period)

        if len(df) >= 2:
            current = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            change_pct = ((current - prev) / prev) * 100
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            return current, change_pct, df
        return 0, 0, pd.DataFrame()
    except Exception:
        return 0, 0, pd.DataFrame()

# --- 3. 차트 생성 함수 ---
def create_chart(df, ticker_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='yellow', width=1), name='MA 5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1.5), name='MA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='purple', width=1.5), name='MA 60'), row=1, col=1)
    
    colors = ['red' if row['Open'] - row['Close'] > 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig

# --- 4. 글로벌 매크로 현황판 ---
st.markdown("### 🌍 Global Market Overview")
with st.spinner("거시 경제 데이터를 불러오는 중..."):
    ndx_p, ndx_c, _ = fetch_data("^NDX", "5d")
    spx_p, spx_c, _ = fetch_data("^GSPC", "5d")
    vix_p, vix_c, _ = fetch_data("^VIX", "5d")
    ks11_p, ks11_c, _ = fetch_data("^KS11", "5d")
    kq11_p, kq11_c, _ = fetch_data("^KQ11", "5d")
    
    oil_p, oil_c, _ = fetch_data("CL=F", "5d")
    btc_p, btc_c, _ = fetch_data("BTC-USD", "5d")
    tnx_p, tnx_c, _ = fetch_data("^TNX", "5d")
    dxy_p, dxy_c, _ = fetch_data("DX-Y.NYB", "5d")
    krw_p, krw_c, _ = fetch_data("KRW=X", "5d")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🇺🇸 나스닥 100", f"{ndx_p:,.2f}", f"{ndx_c:.2f}%")
    c2.metric("🇺🇸 S&P 500", f"{spx_p:,.2f}", f"{spx_c:.2f}%")
    c3.metric("🇰🇷 코스피", f"{ks11_p:,.2f}", f"{ks11_c:.2f}%")
    c4.metric("🇰🇷 코스닥", f"{kq11_p:,.2f}", f"{kq11_c:.2f}%")
    c5.metric("🚨 VIX (공포)", f"{vix_p:,.2f}", f"{vix_c:.2f}%", delta_color="inverse")

    st.markdown("")
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("🛢️ WTI 유가", f"${oil_p:,.2f}", f"{oil_c:.2f}%")
    c7.metric("₿ 비트코인", f"${btc_p:,.0f}", f"{btc_c:.2f}%")
    c8.metric("🇺🇸 10년물 국채", f"{tnx_p:.3f}%", f"{tnx_c:.2f}%")
    c9.metric("💵 달러 인덱스", f"{dxy_p:.2f}", f"{dxy_c:.2f}%")
    c10.metric("🇰🇷 원/달러", f"₩{krw_p:,.2f}", f"{krw_c:.2f}%", delta_color="inverse")

st.markdown("---")

# --- 4-1. 미국 핵심 섹터 자금 흐름 & 히트맵 ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 💸 Sector Fund Flow (미국 증시)")
    with st.spinner("섹터별 수급 데이터를 분석 중입니다..."):
        etf_tickers = {"반도체(SOXX)": "SOXX", "기술주(XLK)": "XLK", "소비재(XLY)": "XLY", "금융(XLF)": "XLF", "헬스케어(XLV)": "XLV", "에너지(XLE)": "XLE"}
        etf_data = []
        for name, tckr in etf_tickers.items():
            _, chg, _ = fetch_data(tckr, "5d")
            etf_data.append({"Sector": name, "Change": chg})
            
        df_etf = pd.DataFrame(etf_data).sort_values(by="Change", ascending=True)
        fig_etf = go.Figure(go.Bar(
            x=df_etf["Change"], y=df_etf["Sector"], orientation='h',
            marker_color=['#FF5252' if val < 0 else '#4CAF50' for val in df_etf["Change"]],
            text=[f"{val:+.2f}%" for val in df_etf["Change"]], textposition='auto', textfont=dict(color='white', size=12, weight='bold')
        ))
        fig_etf.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_etf, use_container_width=True)

with col_right:
    st.markdown("### 🔥 K-Market Heatmap (스캘핑 관심종목)")
    with st.spinner("주도주 변동성을 스캔 중입니다..."):
        kr_watch_list = {
            "삼성전자": "005930", "SK하이닉스": "000660", "한미반도체": "042700",
            "에코프로비엠": "247540", "에코프로": "086520", "엔켐": "348370",
            "HLB": "028300", "알테오젠": "196170", "셀트리온": "068270",
            "현대차": "005380", "KB금융": "105560", "NAVER": "035420"
        }
        hm_data = []
        for name, code in kr_watch_list.items():
            _, chg_p, _ = fetch_data(code, "5d")
            hm_data.append({"Name": name, "Change": chg_p, "Size": 1})
            
        df_hm = pd.DataFrame(hm_data)
        fig_hm = px.treemap(
            df_hm, path=['Name'], values='Size', color='Change',
            color_continuous_scale=['#FF5252', '#21262D', '#4CAF50'], 
            color_continuous_midpoint=0
        )
        fig_hm.update_traces(texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%", customdata=df_hm[['Change']], textfont=dict(size=14, color='white'))
        fig_hm.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
        st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")

# --- 5. 탭 메뉴 (미국장 / 한국장) ---
tab_us, tab_kr = st.tabs(["🇺🇸 미국 시장 (US Market)", "🇰🇷 국내 시장 (KR Market)"])

with tab_us:
    col_side, col_main = st.columns([1, 3])
    with col_side:
        st.markdown("### ⚙️ Analysis Setting")
        us_ticker = st.text_input("티커 입력 (예: TSLA)", value="TSLA")
        period = st.selectbox("기간 선택", ["1mo", "3mo", "6mo", "1y"], index=1)
        api_key_us = st.text_input("Gemini API Key", type="password", key="us_api")
        analyze_btn_us = st.button("🚀 분석 시작", key="us_btn", use_container_width=True)
        
    with col_main:
        if analyze_btn_us and us_ticker:
            if not api_key_us: st.error("⚠️ 왼쪽에 API 키를 입력해주세요!")
            else:
                with st.spinner("분석 중..."):
                    curr, chg, df = fetch_data(us_ticker, period)
                    if not df.empty:
                        st.markdown(f"<div class='big-font'>{us_ticker} <span style='color:{'#FF5252' if chg < 0 else '#4CAF50'};'>({chg:+.2f}%)</span></div>", unsafe_allow_html=True)
                        st.plotly_chart(create_chart(df, us_ticker), use_container_width=True)

                        genai.configure(api_key=api_key_us)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        prompt = f"""대상: {us_ticker} / 현재가: ${curr:,.2f} / 흐름: {df['Close'].tail(5).tolist()}
                        이 데이터를 바탕으로 오늘 장 초반 스캘핑이나 종가배팅 전략을 3줄로 요약해. (진입, 손절, 목표가 필수)"""
                        
                        st.success(model.generate_content(prompt).text)

with tab_kr:
    col_side_kr, col_main_kr = st.columns([1, 3])
    with col_side_kr:
        st.markdown("### ⚙️ 분석 설정")
        kr_ticker = st.text_input("종목코드 (예: 005930)", value="005930")
        period_kr = st.selectbox("조회 기간", ["1mo", "3mo", "6mo", "1y"], index=1, key="kr_period")
        api_key_kr = st.text_input("Gemini API 키", type="password", key="kr_api")
        analyze_btn_kr = st.button("🚀 분석 시작", key="kr_btn", use_container_width=True)

    with col_main_kr:
        if analyze_btn_kr and kr_ticker:
            if not api_key_kr: st.error("⚠️ 왼쪽에 API 키를 입력해주세요!")
            else:
                with st.spinner("분석 중..."):
                    curr_kr, chg_kr, df_kr = fetch_data(kr_ticker, period_kr)
                    if not df_kr.empty:
                        st.markdown(f"<div class='big-font'>{kr_ticker} <span style='color:{'#FF5252' if chg_kr < 0 else '#4CAF50'};'>({chg_kr:+.2f}%)</span></div>", unsafe_allow_html=True)
                        st.plotly_chart(create_chart(df_kr, kr_ticker), use_container_width=True)

                        genai.configure(api_key=api_key_kr)
                        model_kr = genai.GenerativeModel('gemini-2.5-flash')

                        prompt_kr = f"""대상: {kr_ticker} / 현재가: ₩{curr_kr:,.0f} / 흐름: {df_kr['Close'].tail(5).tolist()}
                        이 데이터를 바탕으로 오늘 장 초반 스캘핑이나 종가배팅 전략을 3줄로 요약해. (진입, 손절, 목표가 필수)"""
                        
                        st.success(model_kr.generate_content(prompt_kr).text)
