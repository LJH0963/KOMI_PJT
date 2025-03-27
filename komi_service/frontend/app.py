import streamlit as st
import sys
import os

# 현재 스크립트의 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.config import setup_page
from frontend.session import init_session_state, app_state
from frontend.pages import exercise_select_page, exercise_view_page

def main():
    """메인 앱 함수"""
    try:
        # 페이지 설정
        setup_page()
        
        # 세션 상태 초기화
        init_session_state()
        
        # 페이지 라우팅
        if st.session_state.page == 'exercise_select':
            exercise_select_page()
        elif st.session_state.page == 'exercise_view':
            exercise_view_page()
        else:
            exercise_select_page()
            
    except Exception as e:
        st.error("앱 실행 오류가 발생했습니다. 페이지를 새로 고침해주세요.")
        print(f"오류 발생: {str(e)}")
    finally:
        # 종료 플래그 설정 (앱이 종료될 때 스레드가 정리되도록)
        app_state.is_running = False

if __name__ == "__main__":
    main() 