import streamlit as st
from typing import List, Dict, Any

# 세션 상태 초기화
def init_session_state():
    """스트림릿 세션 상태 초기화"""
    if 'selected_cameras' not in st.session_state:
        st.session_state.selected_cameras = []
    if 'cameras' not in st.session_state:
        st.session_state.cameras = []
    if 'server_status' not in st.session_state:
        st.session_state.server_status = None
    if 'camera_images' not in st.session_state:
        st.session_state.camera_images = {}
    if 'image_update_time' not in st.session_state:
        st.session_state.image_update_time = {}
    if 'sync_status' not in st.session_state:
        st.session_state.sync_status = "준비 중..."
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = {}
    if 'pose_data' not in st.session_state:
        st.session_state.pose_data = {}
    if 'show_pose_overlay' not in st.session_state:
        st.session_state.show_pose_overlay = True
    if 'page' not in st.session_state:
        st.session_state.page = 'exercise_select'
    if 'selected_exercise' not in st.session_state:
        st.session_state.selected_exercise = None
    if 'front_camera' not in st.session_state:
        st.session_state.front_camera = "선택 안함"
    if 'side_camera' not in st.session_state:
        st.session_state.side_camera = "선택 안함"
    if 'show_videos' not in st.session_state:
        st.session_state.show_videos = False
    if 'thread_started' not in st.session_state:
        st.session_state.thread_started = False

# 페이지 이동 함수들
def go_to_exercise_view():
    """운동하기 페이지로 이동"""
    st.session_state.page = 'exercise_view'
    
def go_to_exercise_select():
    """운동 선택 페이지로 이동"""
    st.session_state.page = 'exercise_select'
    st.session_state.show_videos = False

# 글로벌 상태 변수를 위한 상태 클래스
class AppState:
    """전역 상태 관리 클래스"""
    def __init__(self):
        # 서버 시간 동기화 관련 변수
        self.server_time_offset = 0.0  # 서버와의 시간차 (초 단위)
        self.last_time_sync = 0  # 마지막 시간 동기화 시간
        
        # 스레드 안전 데이터 구조
        self.image_queues = {}  # 카메라별 이미지 큐
        self.is_running = True
        self.selected_cameras = []
        
        # 연결 재시도 설정
        self.connection_attempts = {}  # 카메라ID -> 시도 횟수
        
        # 동기화 버퍼 설정
        self.sync_buffer = {}  # 카메라 ID -> 최근 프레임 버퍼
        
        # WebSocket 연결 상태
        self.ws_connection_status = {}  # 카메라ID -> 상태 ("connected", "disconnected", "reconnecting")
        
        # 포즈 데이터 저장소
        self.pose_data_store = {}  # 카메라ID -> 최신 포즈 데이터
        self.pose_update_times = {}  # 카메라ID -> 마지막 포즈 업데이트 시간

# 전역 상태 객체 생성
app_state = AppState() 