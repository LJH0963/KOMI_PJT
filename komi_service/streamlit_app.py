import streamlit as st
import requests
import json
import time
import numpy as np
import cv2
import base64
from datetime import datetime

# 서버 URL 설정
API_URL = "http://localhost:8000"

# 갱신 간격 (초)
REFRESH_INTERVAL = 0.3

# 세션 상태 초기화
if 'selected_camera' not in st.session_state:
    st.session_state.selected_camera = None
if 'cameras' not in st.session_state:
    st.session_state.cameras = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Base64 이미지 디코딩 함수
def decode_image(base64_image):
    """Base64 인코딩된 이미지를 디코딩하여 numpy 배열로 변환"""
    try:
        img_data = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"이미지 디코딩 오류: {str(e)}")
        return None

# 카메라 목록 가져오기
def get_cameras():
    try:
        response = requests.get(f"{API_URL}/cameras")
        if response.status_code == 200:
            data = response.json()
            return data.get("cameras", [])
        return []
    except Exception as e:
        st.error(f"카메라 목록 요청 오류: {str(e)}")
        return []

# 카메라 이미지 요청
def get_camera_image(camera_id):
    try:
        response = requests.get(f"{API_URL}/latest_image/{camera_id}")
        if response.status_code == 200:
            data = response.json()
            return data.get("image_data"), data.get("pose_data")
        return None, None
    except Exception as e:
        return None, None

# 이미지 업데이트 함수
def update_image():
    if st.session_state.selected_camera:
        image_data, _ = get_camera_image(st.session_state.selected_camera)
        if image_data:
            image = decode_image(image_data)
            if image is not None:
                st.session_state.current_image = image
                st.session_state.last_update = time.time()

# 메인 UI
def main():
    st.set_page_config(page_title="KOMI 모니터링", layout="wide")
    
    # 상단 헤더
    st.title("KOMI 웹캠 모니터링")
    
    # 카메라 목록이 없으면 가져오기
    if not st.session_state.cameras:
        st.session_state.cameras = get_cameras()
    
    # 첫 번째 카메라 자동 선택
    if not st.session_state.selected_camera and st.session_state.cameras:
        st.session_state.selected_camera = st.session_state.cameras[0]
    
    # 메인 컨텐츠 영역
    if st.session_state.selected_camera:
        # 현재 시간 기준으로 이미지 업데이트가 필요한지 확인
        current_time = time.time()
        if current_time - st.session_state.last_update >= REFRESH_INTERVAL:
            update_image()
        
        # 이미지 표시 영역
        image_placeholder = st.empty()
        
        # 이미지 표시
        if st.session_state.current_image is not None:
            image_placeholder.image(
                st.session_state.current_image, 
                caption=f"카메라: {st.session_state.selected_camera}", 
                use_container_width=True
            )
        else:
            image_placeholder.info("이미지 로딩 중...")
        
        # 상태 표시
        status_container = st.empty()
        status_container.text(f"마지막 업데이트: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        # 짧은 대기 후 페이지 갱신 (부드러운 업데이트를 위해)
        time.sleep(0.1)
        st.rerun()
    else:
        st.info("사용 가능한 카메라가 없습니다.")

if __name__ == "__main__":
    main() 