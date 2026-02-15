import streamlit as st
import FinanceDataReader as fdr
import google.generativeai as genai
import datetime

# 1. 화면 기본 설정
st.set_page_config(page_title="AI 주식 비서", layout="wide")
st.title("📈 내 손안의 AI 주식 비서 (스캘핑/종가배팅 타점 분석)")

# 2. 왼쪽 사이드바 설정 (API 키 및 종목코드 입력란)
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("발급받은 Gemini API 키 입력", type="password")
    ticker = st.text_input("종목코드 6자리 (예: 삼성전자 005930)", value="005930")
    analyze_btn = st.button("🚀 AI 분석 시작")

# 3. 분석 버튼을 눌렀을 때 작동하는 로직
if analyze_btn:
    if not api_key:
        st.warning("👈 왼쪽에 API 키를 먼저 입력해주세요!")
    else:
        try:
            # 최근 3개월 데이터 가져오기
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=90)
            df = fdr.DataReader(ticker, start_date, end_date)
            
            # 화면에 차트 그리기
            st.subheader(f"📊 {ticker} 최근 3개월 주가 흐름")
            st.line_chart(df['Close'])

            # AI에게 분석 요청하기
            genai.configure(api_key=api_key)
            # 무료 버전인 flash 모델 사용
            model = genai.GenerativeModel('gemini-2.5-flash') 
            
            # AI에게 내릴 명령서 (프롬프트)
            prompt = f"""
            너는 단기매매 전문가야. 
            다음은 {ticker} 종목의 최근 10일간 종가 데이터야:
            {df['Close'].tail(10).to_string()}
            
            이 데이터를 바탕으로 오늘 종가배팅을 들어가거나 내일 장초반 스캘핑을 할 때 
            주의할 점과 접근 전략을 딱 3줄로 핵심만 요약해줘.
            """
            
            with st.spinner("AI가 차트를 분석 중입니다. 잠시만 기다려주세요..."):
                response = model.generate_content(prompt)
                st.success("✨ 분석 완료!")
                st.info(response.text)

        except Exception as e:
            st.error(f"오류가 발생했습니다. 종목코드를 다시 확인해주세요! (에러내용: {e})")