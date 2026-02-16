import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
import google.generativeai as genai
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import requests
import io

# --- 1. 화면 기본 설정 및 디자인 ---
st.set_page_config(page_title="Pro AI Trader", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; font-family: 'Roboto', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    div[data-testid="metric-container"] { background-color: #21262D; border: 1px solid #30363D; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stTabs [data-baseweb="tab"] { background-color: #161B22; border-radius: 5px 5px 0 0; color: #8B949E; font-weight: 600; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    div.stButton > button { background-color: #238636; color: white; border: none; font-weight: bold; width: 100%; height: 45px; }
    div.stButton > button:hover { background-color: #2EA043; }
    .big-font { font-size: 1.5rem !important; font-weight: bold; }
    .score-box { padding: 20px; background-color: #161B22; border-radius: 10px; border: 1px solid #30363D; line-height: 1.6; }
    div.row-widget.stRadio > div { flex-direction: row; gap: 20px; }
    .news-link { color: #58A6FF; text-decoration: none; font-weight: 500; display: block; margin-bottom: 8px; font-size: 14px;}
    .news-link:hover { text-decoration: underline; color: #79C0FF; }
    .dataframe { width: 100% !important; text-align: center !important; color: white; }
    .dataframe th { background-color: #30363D; text-align: center !important; padding: 10px; }
    .dataframe td { padding: 10px; border-bottom: 1px solid #30363D; }
</style>
""", unsafe_allow_html=True)

# --- 2. 테마별 바스켓 종목 설정 ---
THEMES_KR = {
    "🔥 반도체 / HBM": ["005930", "000660", "042700", "058470", "200710", "036540", "252990"],
    "🔋 2차전지": ["373220", "247540", "086520", "003670", "051910", "348370"],
    "💊 제약 / 바이오": ["207940", "068270", "196170", "028300", "068240", "128940"],
    "🚗 자동차 / 로봇": ["005380", "000270", "277810", "056190", "028150", "010140"],
    "💼 금융 / 저PBR": ["105560", "055550", "316140", "032830", "086790", "024110"]
}
THEMES_US = {
    "🤖 AI / 반도체": ["NVDA", "AMD", "TSM", "AVGO", "ASML", "INTC", "QCOM", "MU"],
    "🍎 Big Tech (M7)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX"],
    "💊 헬스케어 / 비만약": ["LLY", "NVO", "MRK", "ABBV", "AMGN", "PFE"],
    "⚡ EV / 친환경": ["TSLA", "RIVN", "LCID", "ENPH", "FSLR", "ALB"],
    "💳 금융 / 핀테크": ["JPM", "V", "MA", "BAC", "PYPL", "SQ"]
}

# --- 3. 사이드바 ---
with st.sidebar:
    st.markdown("## 🚨 Risk & Macro Calendar")
    events = {
        "날짜": ["02/18 (수)", "02/26 (목)", "03/06 (금)", "03/11 (수)", "03/18 (수)"],
        "이벤트": ["FOMC 의사록 공개", "미국 PCE물가지수", "미국 고용보고서", "미국 CPI 발표", "FOMC 금리 결정"],
        "중요도": ["⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
    }
    st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True)
    st.markdown("---")
    st.markdown("### ✅ 트레이딩 Check List")
    st.checkbox("🇺🇸 미국장 나스닥/반도체 흐름 확인")
    st.checkbox("🚨 VIX(공포지수) 안정권 확인")
    st.checkbox("🔥 국내 주도주 외인/기관 양매수 확인")
    st.checkbox("🌙 오늘 밤 주요 매크로 발표 체크")

st.title("🚀 Pro AI Trading Dashboard")
st.markdown("---")

# --- 4. 데이터 수집 엔진 ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Referer': 'https://finance.naver.com/'
}

@st.cache_data(ttl=86400) 
def get_kr_info(search_text):
    try:
        df_krx = fdr.StockListing('KRX')
        if search_text.isdigit():
            row = df_krx[df_krx['Code'] == search_text]
            if not row.empty: return search_text, row['Name'].values[0]
        else:
            row = df_krx[df_krx['Name'] == search_text]
            if not row.empty: return row['Code'].values[0], search_text
    except: pass
    return search_text, search_text

@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y"):
    try:
        if ticker.startswith("^") or ticker.endswith("=X") or ticker.endswith("=F") or ticker.endswith("NYB"):
             df = yf.Ticker(ticker).history(period=period)
        elif ticker.isdigit():
             end_date = datetime.date.today(); start_date = end_date - datetime.timedelta(days=730)
             df = fdr.DataReader(ticker, start_date, end_date)
             df = df.rename(columns={'Close': 'Close', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume'})
             df.index.name = 'Date'
             df = df.loc[df.index >= (pd.Timestamp.now() - pd.Timedelta(days=365 if period == "1y" else 730))]
        else:
             df = yf.Ticker(ticker).history(period=period)

        if len(df) >= 20:
            current = df['Close'].iloc[-1]; prev = df['Close'].iloc[-2]; change_pct = ((current - prev) / prev) * 100
            
            df['MA5'] = df['Close'].rolling(window=5).mean(); df['MA20'] = df['Close'].rolling(window=20).mean(); df['MA60'] = df['Close'].rolling(window=60).mean()
            df['Env_Up'] = df['MA20'] * 1.10; df['Env_Down'] = df['MA20'] * 0.90
            delta = df['Close'].diff(); gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean(); loss = -1 * delta.clip(upper=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))
            exp1 = df['Close'].ewm(span=12, adjust=False).mean(); exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2; df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean(); df['MACD_Hist'] = df['MACD'] - df['Signal']
            return current, change_pct, df
        return 0, 0, pd.DataFrame()
    except: return 0, 0, pd.DataFrame()

@st.cache_data(ttl=300)
def get_heatmap_pct(ticker, is_kr=False):
    try:
        if is_kr:
            df = fdr.DataReader(ticker, (pd.Timestamp.now() - pd.Timedelta(days=14)).strftime('%Y-%m-%d'))
        else:
            df = yf.Ticker(ticker).history(period="1mo")
            
        if len(df) >= 2:
            df = df.dropna(subset=['Close'])
            prev_c = df['Close'].iloc[-2]; curr_c = df['Close'].iloc[-1]
            if prev_c > 0:
                return round(((curr_c - prev_c) / prev_c) * 100, 2)
    except: pass
    return 0.00

@st.cache_data(ttl=60) 
def fetch_kr_news(ticker_code):
    try:
        url = f"https://m.stock.naver.com/api/news/stock/{ticker_code}?pageSize=5"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        data = res.json()
        if data: return [{'title': item['tit'], 'link': f"https://n.news.naver.com/mnews/article/{item['oid']}/{item['aid']}"} for item in data]
    except: pass
    return [{'title': "뉴스를 불러오지 못했습니다.", 'link': "#"}]

@st.cache_data(ttl=300)
def fetch_investor_data(ticker_code):
    try:
        url = f"https://m.stock.naver.com/api/stock/{ticker_code}/investor/days?pageSize=5&page=1"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        data = res.json()
        parsed = []
        for row in data:
            parsed.append({
                '날짜': row['bizdate'],
                '기관 (주)': int(row.get('investorInstitutionExact', 0)),
                '외국인 (주)': int(row.get('investorForeignExact', 0))
            })
        if parsed:
            return pd.DataFrame(parsed)
    except: pass
    return pd.DataFrame(columns=['날짜', '기관 (주)', '외국인 (주)'])

def calculate_scores(df, current, df_investor=None):
    sc_end, sc_sw, sc_lg = 50, 50, 50
    try:
        ma5 = df['MA5'].iloc[-1]; ma20 = df['MA20'].iloc[-1]; ma60 = df['MA60'].iloc[-1]
        vol_mean = df['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = df['Volume'].iloc[-1] / vol_mean if vol_mean > 0 else 1
        rsi = df['RSI'].iloc[-1]; env_down = df['Env_Down'].iloc[-1]; macd_hist = df['MACD_Hist'].iloc[-1]

        supply_bonus = 0
        if df_investor is not None and not df_investor.empty:
            i_net = df_investor.iloc[0]['기관 (주)']; f_net = df_investor.iloc[0]['외국인 (주)']
            if i_net > 0 and f_net > 0: supply_bonus += 20
            elif i_net > 0 or f_net > 0: supply_bonus += 10
            elif i_net < 0 and f_net < 0: supply_bonus -= 15

        sc_end = 40 + supply_bonus
        if vol_ratio > 1.5: sc_end += 20 
        if current > ma5: sc_end += 10 
        if current <= env_down * 1.02: sc_end += 20 
        if rsi < 35: sc_end += 10 
        elif rsi > 70: sc_end -= 20 
        
        sc_sw = 40 + (supply_bonus * 0.7)
        if ma5 > ma20: sc_sw += 15 
        if current > ma20: sc_sw += 15 
        if macd_hist > 0: sc_sw += 20 
        if 40 <= rsi <= 65: sc_sw += 10 

        sc_lg = 40 + (supply_bonus * 0.5)
        if ma5 > ma20 > ma60: sc_lg += 30 
        if current > ma60: sc_lg += 20 
        if macd_hist > 0: sc_lg += 10
        if rsi < 50: sc_lg += 10 
    except: pass

    return {"종가배팅": min(max(int(sc_end), 0), 100), "스윙매매": min(max(int(sc_sw), 0), 100), "장기투자": min(max(int(sc_lg), 0), 100)}

# --- 차트 생성 함수들 ---
def create_horizontal_candle(score):
    color = "#FF5252" if score < 40 else "#FFA726" if score < 70 else "#4CAF50"
    fig = go.Figure(go.Indicator(
        mode = "number+gauge", value = score,
        number = {'suffix': "점", 'font': {'size': 20, 'color': color}},
        gauge = {
            'shape': "bullet", 'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': color, 'thickness': 0.8}, 'bgcolor': "#30363D",
            'steps': [{'range': [0, 100], 'color': "rgba(0,0,0,0)"}],
        }
    ))
    fig.update_layout(height=40, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def create_gauge(score, style_name):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': f"🎯 {style_name} 매력도", 'font': {'color': 'white', 'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"}, 'bar': {'color': "#E0E0E0"}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "#30363D",
            'steps': [{'range': [0, 40], 'color': '#FF5252'}, {'range': [40, 70], 'color': '#FFA726'}, {'range': [70, 100], 'color': '#4CAF50'}],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': score}
        }))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    return fig

def create_main_chart(df, ticker_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    if 'MA5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='yellow', width=1), name='MA 5'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1.5), name='MA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='purple', width=1.5), name='MA 60'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Env_Up'], line=dict(color='rgba(0, 255, 255, 0.5)', width=1, dash='dot'), name='Env 상단 (+10%)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Env_Down'], line=dict(color='rgba(0, 255, 255, 0.5)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.05)', name='Env 하단 (-10%)'), row=1, col=1)
    colors = ['red' if row['Open'] - row['Close'] > 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, title=f"📊 {ticker_name} 일봉 & 엔벨로프 차트")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig

def create_sub_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.5])
    if 'MACD_Hist' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=['rgba(76, 175, 80, 0.7)' if val >= 0 else 'rgba(255, 82, 82, 0.7)' for val in df['MACD_Hist']], name='MACD Hist'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2196F3', width=1.5), name='MACD'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='#FF9800', width=1.5), name='Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#E040FB', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 82, 82, 0.5)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(76, 175, 80, 0.5)", row=2, col=1)
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, title="📉 모멘텀 지표 (MACD & RSI)")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig

# --- 5. 글로벌 매크로 & 분할 히트맵 ---
st.markdown("### 🌍 Global Macro & Market Heatmap")
ndx_p, ndx_c, _ = fetch_data("^NDX", "1mo")
spx_p, spx_c, _ = fetch_data("^GSPC", "1mo")
ks11_p, ks11_c, _ = fetch_data("^KS11", "1mo")
kq11_p, kq11_c, _ = fetch_data("^KQ11", "1mo")
vix_p, vix_c, _ = fetch_data("^VIX", "1mo")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🇺🇸 나스닥 100", f"{ndx_p:,.2f}", f"{ndx_c:.2f}%")
c2.metric("🇺🇸 S&P 500", f"{spx_p:,.2f}", f"{spx_c:.2f}%")
c3.metric("🇰🇷 코스피", f"{ks11_p:,.2f}", f"{ks11_c:.2f}%")
c4.metric("🇰🇷 코스닥", f"{kq11_p:,.2f}", f"{kq11_c:.2f}%")
c5.metric("🚨 VIX (공포지수)", f"{vix_p:,.2f}", f"{vix_c:.2f}%", delta_color="inverse")

col_hm_us, col_hm_kr = st.columns(2)
with col_hm_us:
    st.markdown("#### 🇺🇸 US Tech & Market Cap Heatmap")
    us_top = {"AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "GOOGL": "GOOGL", "AMZN": "AMZN", "META": "META", "TSLA": "TSLA", "LLY": "LLY", "AVGO": "AVGO", "JPM": "JPM"}
    df_us_hm = pd.DataFrame([{"Name": k, "Change": get_heatmap_pct(v, False), "Size": 1} for k, v in us_top.items()])
    fig_us = px.treemap(df_us_hm, path=['Name'], values='Size', color='Change', color_continuous_scale=[(0, '#FF5252'), (0.5, '#21262D'), (1, '#4CAF50')], color_continuous_midpoint=0)
    fig_us.update_traces(texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}%", customdata=df_us_hm[['Change']], textfont=dict(size=16, color='white'))
    fig_us.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_us, use_container_width=True)

with col_hm_kr:
    st.markdown("#### 🇰🇷 KR Top Market Cap Heatmap")
    kr_top = {"삼성전자": "005930", "SK하이닉스": "000660", "LG엔솔": "373220", "삼바": "207940", "현대차": "005380", "기아": "000270", "셀트리온": "068270", "POSCO홀딩스": "005490", "KB금융": "105560", "NAVER": "035420"}
    df_kr_hm = pd.DataFrame([{"Name": k, "Change": get_heatmap_pct(v, True), "Size": 1} for k, v in kr_top.items()])
    fig_kr = px.treemap(df_kr_hm, path=['Name'], values='Size', color='Change', color_continuous_scale=[(0, '#FF5252'), (0.5, '#21262D'), (1, '#4CAF50')], color_continuous_midpoint=0)
    fig_kr.update_traces(texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}%", customdata=df_kr_hm[['Change']], textfont=dict(size=16, color='white'))
    fig_kr.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_kr, use_container_width=True)

st.markdown("---")

# --- 6. 탭 시스템 ---
tab_us, tab_kr, tab_theme, tab_port = st.tabs(["🇺🇸 미국 시장", "🇰🇷 국내 시장", "🏆 AI 테마 추천 (Top 5)", "💼 내 포트폴리오"])

with tab_us:
    col_s1, col_m1 = st.columns([1, 4])
    with col_s1:
        st.markdown("### ⚙️ Analysis Setting")
        us_ticker = st.text_input("티커 (US)", value="TSLA")
        trade_style_us = st.radio("포지션 선택", ["종가배팅", "스윙매매", "장기투자"], index=0, key="us_style")
        api_key_us = st.text_input("Gemini API Key", type="password", key="us_api")
        btn_us = st.button("🚀 맞춤형 분석 시작", key="us_btn", use_container_width=True)
        
    with col_m1:
        if btn_us and us_ticker:
            with st.spinner(f"{trade_style_us} 관점에서 차트를 분석 중입니다..."):
                curr, chg, df = fetch_data(us_ticker, "1y")
                selected_score = calculate_scores(df, curr)[trade_style_us]

                try: 
                    us_news_data = yf.Ticker(us_ticker).news[:3]
                    us_news_titles = [item['title'] for item in us_news_data]
                    us_news_links = [item['link'] for item in us_news_data]
                except: 
                    us_news_titles = ["최신 뉴스 데이터를 불러올 수 없습니다."]
                    us_news_links = ["#"]

                st.markdown(f"<div class='big-font'>{us_ticker} <span style='color:{'#FF5252' if chg < 0 else '#4CAF50'};'>({chg:+.2f}%)</span></div>", unsafe_allow_html=True)
                st.plotly_chart(create_main_chart(df, us_ticker), use_container_width=True)
                st.plotly_chart(create_sub_chart(df), use_container_width=True)
                
                c_gauge, c_report = st.columns([1, 2])
                with c_gauge: 
                    st.plotly_chart(create_gauge(selected_score, trade_style_us), use_container_width=True)
                    st.markdown("#### 📰 실시간 뉴스 헤드라인")
                    for i, title in enumerate(us_news_titles):
                        st.markdown(f"<a href='{us_news_links[i]}' target='_blank' class='news-link'>• {title}</a>", unsafe_allow_html=True)
                        
                with c_report:
                    st.markdown("#### 🤖 AI 전략 브리핑 (Technical Focus)")
                    if api_key_us:
                        try:
                            genai.configure(api_key=api_key_us)
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            
                            # [핵심 업데이트] 미국장: 뉴스를 빼고 철저한 기술적/전략적 분석만 요구
                            prompt_us = f"""
                            종목명: {us_ticker}
                            현재가: ${curr:,.2f}
                            매매 포지션: {trade_style_us} (AI 매력도: {selected_score}점)
                            최근 5일 종가: {df['Close'].tail(5).tolist()}
                            RSI(14): {df['RSI'].iloc[-1]:.1f}
                            MACD 히스토그램: {df['MACD_Hist'].iloc[-1]:.2f}
                            
                            위 기술적 지표들을 바탕으로 '{trade_style_us}' 관점에 최적화된 고도화된 매매 전략을 작성해줘.
                            다음 양식을 반드시 지켜서 전문가처럼 브리핑해:
                            1. 📊 **차트 및 지표 분석**: (이평선, RSI, MACD 등을 종합하여 현재 추세 2~3줄 요약)
                            2. 🎯 **{trade_style_us} 타점 전략**: (구체적인 진입가, 손절가, 단기/스윙 목표가를 수치로 제시)
                            3. 💡 **리스크 관리 및 비중**: (이 포지션에서 주의할 점과 추천 진입 비중)
                            """
                            
                            res_us = model.generate_content(prompt_us)
                            st.markdown("<div class='score-box'>", unsafe_allow_html=True)
                            st.success(res_us.text)
                            st.markdown("</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error("⚠️ API 키가 유효하지 않거나 한도를 초과했습니다. 다시 확인해 주세요.")
                    else:
                        st.warning("⚠️ API 키를 입력하시면 AI 브리핑을 보실 수 있습니다.")

with tab_kr:
    col_s2, col_m2 = st.columns([1, 4])
    with col_s2:
        st.markdown("### ⚙️ 분석 설정")
        kr_input = st.text_input("종목명 또는 코드 (KR)", value="삼성전자") 
        trade_style_kr = st.radio("포지션 선택", ["종가배팅", "스윙매매", "장기투자"], index=0, key="kr_style")
        api_key_kr = st.text_input("Gemini API Key", type="password", key="kr_api")
        btn_kr = st.button("🚀 맞춤형 분석 시작", key="kr_btn", use_container_width=True)

    with col_m2:
        if btn_kr and kr_input:
            with st.spinner(f"[{kr_input}] 수급 스캔 및 데이터 분석 중..."):
                kr_code, kr_name = get_kr_info(kr_input)
                curr_kr, chg_kr, df_kr = fetch_data(kr_code, "1y")
                
                df_investor = fetch_investor_data(kr_code)
                kr_news_data = fetch_kr_news(kr_code)
                kr_news_titles = [item['title'] for item in kr_news_data]
                
                if df_kr.empty: st.error("데이터를 찾을 수 없습니다.")
                else:
                    display_title = f"{kr_name} ({kr_code})" if kr_name != kr_code else kr_code
                    selected_score_kr = calculate_scores(df_kr, curr_kr, df_investor)[trade_style_kr]
                    
                    st.markdown(f"<div class='big-font'>{display_title} <span style='color:{'#FF5252' if chg_kr < 0 else '#4CAF50'};'>({chg_kr:+.2f}%)</span></div>", unsafe_allow_html=True)
                    st.plotly_chart(create_main_chart(df_kr, display_title), use_container_width=True)
                    st.plotly_chart(create_sub_chart(df_kr), use_container_width=True)
                    
                    c_left, c_mid, c_right = st.columns([1.2, 1, 1.5])
                    with c_left: 
                        st.plotly_chart(create_gauge(selected_score_kr, trade_style_kr), use_container_width=True)
                    with c_mid:
                        st.markdown("#### 💰 최근 5일 매매동향")
                        if not df_investor.empty:
                            def color_net_buy(val): return 'color: #FF5252' if val > 0 else 'color: #4CAF50' if val < 0 else 'color: white'
                            st.dataframe(df_investor.style.map(color_net_buy, subset=['기관 (주)', '외국인 (주)']).format({'기관 (주)': '{:,}', '외국인 (주)': '{:,}'}), hide_index=True, use_container_width=True)
                        else: st.info("수급 데이터를 불러오지 못했습니다.")
                            
                    with c_right:
                        st.markdown("#### 📰 특징주 뉴스")
                        for news in kr_news_data: st.markdown(f"<a href='{news['link']}' target='_blank' class='news-link'>• {news['title']}</a>", unsafe_allow_html=True)
                        
                        st.markdown("#### 🤖 AI 전략 브리핑 (Technical Focus)")
                        if api_key_kr:
                            try:
                                genai.configure(api_key=api_key_kr)
                                model_kr = genai.GenerativeModel('gemini-2.5-flash')
                                
                                # [핵심 업데이트] 한국장: 뉴스/수급 요약을 빼고 철저한 기술적/전략적 분석만 요구
                                prompt_kr = f"""
                                종목명: {kr_name}
                                현재가: ₩{curr_kr:,.0f}
                                매매 포지션: {trade_style_kr} (AI 매력도: {selected_score_kr}점)
                                최근 5일 종가: {df_kr['Close'].tail(5).tolist()}
                                RSI(14): {df_kr['RSI'].iloc[-1]:.1f}
                                MACD 히스토그램: {df_kr['MACD_Hist'].iloc[-1]:.2f}
                                엔벨로프 하단(-10%): ₩{df_kr['Env_Down'].iloc[-1]:.0f}
                                
                                위 기술적 지표들을 바탕으로 '{trade_style_kr}' 관점에 최적화된 고도화된 매매 전략을 작성해줘.
                                다음 양식을 반드시 지켜서 전문가처럼 브리핑해:
                                1. 📊 **차트 및 보조지표 분석**: (현재 배열 상태, RSI/MACD 모멘텀, 엔벨로프 이격도 등을 종합하여 2~3줄로 상세히 분석)
                                2. 🎯 **{trade_style_kr} 타점 전략**: (구체적인 1차/2차 매수 진입가, 명확한 손절가, 단기/스윙/장기 목표가를 수치로 제시)
                                3. 💡 **리스크 관리 및 비중 조절**: (이 포지션에서 주의할 점과 추천 진입 비중)
                                """
                                
                                res_kr = model.generate_content(prompt_kr)
                                st.markdown("<div class='score-box'>", unsafe_allow_html=True)
                                st.success(res_kr.text)
                                st.markdown("</div>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error("⚠️ API 키가 유효하지 않거나 한도를 초과했습니다. 다시 확인해 주세요.")
                        else:
                            st.warning("⚠️ API 키를 입력하시면 AI 브리핑을 보실 수 있습니다.")

with tab_theme:
    st.markdown("### 🏆 AI 테마별 주도주 스캐너 (Top 5 Picks)")
    st.info("시장을 주도하는 핫한 테마를 고르면, AI가 바스켓 종목들을 실시간으로 스캔하여 최적의 타점을 잡은 TOP 5 종목을 추천합니다.")
    
    col_t1, col_t2 = st.columns([1, 4])
    with col_t1:
        st.markdown("### ⚙️ 스캐너 설정")
        market_choice = st.radio("시장 선택", ["🇰🇷 국내 시장 (KR)", "🇺🇸 미국 시장 (US)"])
        
        if "KR" in market_choice: theme_dict = THEMES_KR
        else: theme_dict = THEMES_US
            
        selected_theme = st.selectbox("스캔할 테마 선택", list(theme_dict.keys()))
        scan_position = st.radio("추천 기준", ["종가배팅", "스윙매매", "장기투자"], index=0, key="scan_pos")
        btn_scan = st.button("🚀 테마 스캔 시작", use_container_width=True)
        
    with col_t2:
        if btn_scan:
            with st.spinner(f"[{selected_theme}] 테마 종목들을 스캔하여 {scan_position} 매력도를 계산 중입니다..."):
                scan_results = []
                for code in theme_dict[selected_theme]:
                    if "KR" in market_choice:
                        name = get_kr_info(code)[1]
                        inv_df = fetch_investor_data(code)
                        currency = "₩"
                        decimals = 0
                    else:
                        name = code
                        inv_df = pd.DataFrame()
                        currency = "$"
                        decimals = 2
                        
                    curr, chg, df = fetch_data(code, "3mo") 
                    if not df.empty:
                        score = calculate_scores(df, curr, inv_df)[scan_position]
                        scan_results.append({"종목명": name, "코드": code, "현재가": curr, "등락률": chg, "점수": score})
                
                scan_results = sorted(scan_results, key=lambda x: x["점수"], reverse=True)
                
                st.markdown(f"#### 🏅 [{selected_theme}] {scan_position} AI 추천 랭킹 TOP 5")
                medals = ["🥇 1위", "🥈 2위", "🥉 3위", "🏅 4위", "🏅 5위"]
                for i, res in enumerate(scan_results[:5]):
                    if i < len(medals):
                        medal = medals[i]
                        price_formatted = f"{currency}{res['현재가']:,.{decimals}f}"
                        color = "#FF5252" if res['등락률'] > 0 else "#4CAF50" if res['등락률'] < 0 else "white"
                        
                        st.markdown(f"""
                        <div class='rank-card'>
                            <h3 style='margin-top:0; margin-bottom:5px;'>{medal} : {res['종목명']} <span style='font-size:18px; color:{color};'>({res['등락률']:+.2f}%)</span></h3>
                            <p style='margin-top:0px; margin-bottom:5px; text-align:right; color:#8B949E; font-weight:bold;'>현재가: {price_formatted}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(create_horizontal_candle(res['점수']), use_container_width=True, key=f"bar_{i}_{res['코드']}")
                
                st.success("✅ 스캔이 완료되었습니다! 위 종목의 이름이나 코드를 좌측 탭에서 검색하여 정밀 분석해 보세요.")

with tab_port:
    st.markdown("### 💼 내 포트폴리오 자산 비중 관리 (Live)")
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame([{"종목명_또는_티커": "삼성전자", "매수단가": 75000, "수량": 100}, {"종목명_또는_티커": "TSLA", "매수단가": 200, "수량": 50}])
    st.info("💡 엑셀처럼 자유롭게 입력하세요. (아래 ➕ 버튼으로 행 추가 가능)")
    edited_df = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True,
        column_config={"종목명_또는_티커": st.column_config.TextColumn("종목명(KR) 또는 티커(US)"), "매수단가": st.column_config.NumberColumn("평균 매수단가"), "수량": st.column_config.NumberColumn("보유 수량")})
    st.session_state.portfolio = edited_df
    
    if st.button("🔄 포트폴리오 실시간 비중 분석", use_container_width=True):
        with st.spinner("시장 데이터를 가져와 실시간 비중을 계산 중입니다..."):
            port_data, tot_inv, tot_cur = [], 0, 0
            for index, row in edited_df.iterrows():
                asset = str(row["종목명_또는_티커"]).strip()
                avg_price = float(row["매수단가"]) if pd.notnull(row["매수단가"]) else 0
                qty = float(row["수량"]) if pd.notnull(row["수량"]) else 0
                if asset and qty > 0:
                    code, name = get_kr_info(asset)
                    curr_price, _, _ = fetch_data(code, "5d")
                    if curr_price == 0: 
                        curr_price, _, _ = fetch_data(asset, "5d"); name = asset.upper()
                    if curr_price > 0:
                        invested = avg_price * qty; current_val = curr_price * qty
                        tot_inv += invested; tot_cur += current_val
                        port_data.append({"종목명": name, "수량": qty, "평가금액": current_val})
            if port_data:
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("💰 총 매수금액", f"{tot_inv:,.0f}")
                c2.metric("📈 총 평가금액", f"{tot_cur:,.0f}", f"{(tot_cur-tot_inv)/tot_inv*100:+.2f}%")
                c3.metric("💵 총 평가손익", f"{tot_cur-tot_inv:,.0f}")
                fig_donut = px.pie(pd.DataFrame(port_data), values='평가금액', names='종목명', hole=0.4, title="🍩 내 자산 비중 (Portfolio Allocation)", color_discrete_sequence=px.colors.sequential.Tealgrn)
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                fig_donut.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450)
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.warning("⚠️ 데이터를 불러오지 못했습니다. 종목명을 확인해 주세요.")
