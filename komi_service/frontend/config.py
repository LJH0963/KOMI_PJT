import streamlit as st

# 서버 URL 설정
API_URL = "http://localhost:8000"

# 시간 동기화 설정
TIME_SYNC_INTERVAL = 300  # 5분으로 간격 설정

# 연결 재시도 설정
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 1.0  # 초 단위

# 동기화 버퍼 설정
SYNC_BUFFER_SIZE = 10  # 각 카메라별 버퍼 크기
MAX_SYNC_DIFF_MS = 100  # 프레임 간 최대 허용 시간 차이 (밀리초)

# 페이지 설정
def setup_page():
    """스트림릿 페이지 설정"""
    # 페이지 설정 - 사이드바 넓이 조정
    st.set_page_config(
        page_title="KOMI 운동 가이드", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )

    # 사이드바 넓이 줄이기
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 220px;
            max-width: 220px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Streamlit의 query parameter를 사용하여 서버 URL을 설정
    if 'server_url' in st.query_params:
        global API_URL
        API_URL = st.query_params['server_url'] 