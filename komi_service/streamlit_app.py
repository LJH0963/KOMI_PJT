import streamlit as st
import requests
import json
import time
import numpy as np
import cv2
import base64
from datetime import datetime
from PIL import Image
from io import BytesIO

# 서버 URL 설정
API_URL = "http://localhost:8000"

# 세션 상태 초기화
if 'selected_camera' not in st.session_state:
    st.session_state.selected_camera = None
if 'cameras' not in st.session_state:
    st.session_state.cameras = []
if 'server_status' not in st.session_state:
    st.session_state.server_status = None

# Base64 이미지 디코딩 함수
def decode_image(base64_image):
    """Base64 인코딩된 이미지를 디코딩하여 numpy 배열로 변환"""
    try:
        img_data = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"이미지 디코딩 오류: {str(e)}")
        return None

# 카메라 목록 가져오기
def get_cameras():
    try:
        response = requests.get(f"{API_URL}/cameras", timeout=2)
        if response.status_code == 200:
            data = response.json()
            st.session_state.server_status = "연결됨"
            return data.get("cameras", [])
        st.session_state.server_status = "오류"
        return []
    except Exception as e:
        st.session_state.server_status = "연결 실패"
        print(f"카메라 목록 요청 오류: {str(e)}")
        return []

# 직접 이미지 데이터 요청
def get_raw_image(camera_id):
    try:
        response = requests.get(f"{API_URL}/get-image/{camera_id}", timeout=0.1)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        print(f"이미지 요청 오류: {str(e)}")
        return None

# JSON 형식의 이미지 요청
def get_camera_image(camera_id):
    try:
        response = requests.get(f"{API_URL}/latest_image/{camera_id}")
        if response.status_code == 200:
            data = response.json()
            return data.get("image_data"), data.get("pose_data")
        return None, None
    except Exception as e:
        print(f"이미지 요청 오류: {str(e)}")
        return None, None

# 메인 UI
def main():
    st.set_page_config(page_title="KOMI 모니터링", layout="wide")
    
    # 상단 헤더
    st.title("KOMI 웹캠 모니터링")
    
    # 서버 상태 확인
    if st.session_state.server_status is None:
        with st.spinner("서버 연결 중..."):
            st.session_state.cameras = get_cameras()
    
    # 서버 상태에 따른 처리
    if st.session_state.server_status == "연결 실패":
        st.error("서버에 연결할 수 없습니다")
        if st.button("재연결"):
            st.session_state.cameras = get_cameras()
            st.rerun()
        return
    
    # 카메라 목록이 없으면 표시 후 종료
    if not st.session_state.cameras:
        st.info("연결된 카메라가 없습니다")
        if st.button("새로고침"):
            st.session_state.cameras = get_cameras()
            st.rerun()
        return
    
    # 첫 번째 카메라 자동 선택
    if not st.session_state.selected_camera and st.session_state.cameras:
        st.session_state.selected_camera = st.session_state.cameras[0]
    
    # 이미지 표시 영역
    image_slot = st.empty()
    status_slot = st.empty()
    
    # 항상 스트리밍 모드로 실행
    status_slot.text("실시간 스트리밍 중...")
    
    # 이미지 스트리밍 루프
    while True:
        try:
            # 방법 1: 바이너리 이미지 직접 요청
            img_data = get_raw_image(st.session_state.selected_camera)
            if img_data:
                img_bytes = BytesIO(img_data)
                image = Image.open(img_bytes)
                image_slot.image(image, use_container_width=True)
            else:
                # 방법 2: JSON API로 대체 시도
                image_data, pose_data = get_camera_image(st.session_state.selected_camera)
                if image_data:
                    image = decode_image(image_data)
                    if image is not None:
                        image_slot.image(image, use_container_width=True)
                else:
                    image_slot.info("이미지를 가져올 수 없습니다")
            
            # 상태 업데이트 (시간만 표시)
            now = datetime.now().strftime('%H:%M:%S')
            status_slot.text(f"업데이트: {now}")
            
        except Exception as e:
            status_slot.error(f"오류: {str(e)}")
        
        # 짧은 대기
        time.sleep(0.1)

if __name__ == "__main__":
    main() 