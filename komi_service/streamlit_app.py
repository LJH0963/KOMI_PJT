import streamlit as st

# 페이지 설정
st.set_page_config(page_title="KOMI 모니터링", layout="wide")

import json
import time
import numpy as np
import cv2
import base64
from datetime import datetime, timedelta
from PIL import Image
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import collections
import random
import requests

# 웹소켓 관련 함수 임포트 - 실제 사용하는 함수만 임포트
from streamlit_websocket import (
    run_async, async_get_cameras, run_async_loop, 
    image_queues, ws_connection_status
)

# 서버 URL 설정
API_URL = "http://localhost:8000"

# Streamlit의 query parameter를 사용하여 서버 URL을 설정
if 'server_url' in st.query_params:
    API_URL = st.query_params['server_url']

# 스레드 안전 데이터 구조
is_running = True
selected_cameras = []

# 서버 시간 동기화 관련 변수
server_time_offset = 0.0  # 서버와의 시간차 (초 단위)
last_time_sync = 0  # 마지막 시간 동기화 시간
TIME_SYNC_INTERVAL = 300  # 5분으로 간격 증가

# 동기화 버퍼 설정
sync_buffer = {}  # 카메라 ID -> 최근 프레임 버퍼 (collections.deque)
SYNC_BUFFER_SIZE = 10  # 각 카메라별 버퍼 크기
MAX_SYNC_DIFF_MS = 100  # 프레임 간 최대 허용 시간 차이 (밀리초)

# 이미지 처리를 위한 스레드 풀
thread_pool = ThreadPoolExecutor(max_workers=4)

# 세션 상태 초기화
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
if 'page' not in st.session_state:
    st.session_state.page = "main"
if 'selected_exercise' not in st.session_state:
    st.session_state.selected_exercise = None
if 'exercises' not in st.session_state:
    st.session_state.exercises = []

# Base64 이미지 디코딩 함수
def decode_image(base64_image):
    """Base64 인코딩된 이미지를 디코딩하여 numpy 배열로 변환"""
    try:
        if not base64_image:
            return None
        img_data = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"이미지 디코딩 오류: {str(e)}")
        return None

# 스레드에서 이미지 디코딩 처리
def process_image_in_thread(image_data):
    """별도 스레드에서 이미지 처리"""
    try:
        if not image_data:
            return None
        return decode_image(image_data)  # Base64 이미지인 경우
    except Exception as e:
        print(f"이미지 처리 오류: {str(e)}")
        return None

# 동기화 버퍼 초기화
def init_sync_buffer(camera_ids):
    """동기화 버퍼 초기화 함수"""
    global sync_buffer
    for camera_id in camera_ids:
        if camera_id not in sync_buffer:
            sync_buffer[camera_id] = collections.deque(maxlen=SYNC_BUFFER_SIZE)

# 동기화된 프레임 쌍 찾기
def find_synchronized_frames():
    """여러 카메라에서 타임스탬프가 가장 가까운 프레임 쌍 찾기"""
    global sync_buffer
    
    if len(st.session_state.selected_cameras) < 2:
        return None
    
    if not all(camera_id in sync_buffer and len(sync_buffer[camera_id]) > 0 
               for camera_id in st.session_state.selected_cameras):
        return None
    
    # 가장 최근의 타임스탬프 기준으로 시작
    latest_frame_times = {
        camera_id: max([frame["time"] for frame in sync_buffer[camera_id]])
        for camera_id in st.session_state.selected_cameras
    }
    
    # 가장 늦은 타임스탬프 찾기
    reference_time = min(latest_frame_times.values())
    
    # 각 카메라에서 기준 시간과 가장 가까운 프레임 찾기
    best_frames = {}
    for camera_id in st.session_state.selected_cameras:
        closest_frame = min(
            [frame for frame in sync_buffer[camera_id]],
            key=lambda x: abs((x["time"] - reference_time).total_seconds())
        )
        
        # 시간 차이가 허용 범위 내인지 확인
        time_diff_ms = abs((closest_frame["time"] - reference_time).total_seconds() * 1000)
        if time_diff_ms <= MAX_SYNC_DIFF_MS:
            best_frames[camera_id] = closest_frame
        else:
            return None
    
    # 모든 카메라에 대해 프레임을 찾았는지 확인
    if len(best_frames) == len(st.session_state.selected_cameras):
        # 동기화 상태 업데이트
        max_diff = max([abs((f["time"] - reference_time).total_seconds() * 1000) 
                        for f in best_frames.values()])
        st.session_state.sync_status = f"동기화됨 (최대 차이: {max_diff:.1f}ms)"
        return best_frames
    
    return None

# 동기 카메라 목록 가져오기 (Streamlit 호환용)
def get_cameras():
    """카메라 목록 가져오기 (동기 래퍼)"""
    cameras, status = run_async(async_get_cameras(API_URL))
    st.session_state.server_status = status
    return cameras

# 운동 목록 가져오기
def get_exercises():
    """운동 목록 가져오기"""
    try:
        response = requests.get(f"{API_URL}/exercises")
        if response.status_code == 200:
            return response.json().get("exercises", [])
        return []
    except Exception as e:
        print(f"운동 목록 요청 오류: {str(e)}")
        return []

# 카메라 UI 업데이트 루프
def update_camera_loop(image_slots, status_slots, connection_indicators, use_sync):
    try:
        update_interval = 0
        while True:
            update_interval += 1
            update_ui = False
            
            if use_sync and len(st.session_state.selected_cameras) > 1:
                # 동기화된 프레임 찾기
                sync_frames = find_synchronized_frames()
                if sync_frames:
                    # 동기화된 프레임이 있으면 UI 업데이트
                    for camera_id, frame_data in sync_frames.items():
                        st.session_state.camera_images[camera_id] = frame_data["image"]
                        st.session_state.image_update_time[camera_id] = frame_data["time"]
                    update_ui = True
            else:
                # 동기화 없이 각 카메라의 최신 프레임 사용
                for camera_id in st.session_state.selected_cameras[:2]:
                    if camera_id in image_queues and not image_queues[camera_id].empty():
                        try:
                            img_data = image_queues[camera_id].get(block=False)
                            st.session_state.camera_images[camera_id] = img_data.get("image")
                            st.session_state.image_update_time[camera_id] = img_data.get("time")
                            update_ui = True
                        except queue.Empty:
                            pass
            
            # 이미지 업데이트
            if update_ui:
                for camera_id in st.session_state.selected_cameras[:2]:
                    if camera_id in st.session_state.camera_images:
                        img = st.session_state.camera_images[camera_id]
                        if img is not None:
                            image_slots[camera_id].image(img, use_container_width=True)
                            status_time = st.session_state.image_update_time[camera_id].strftime('%H:%M:%S.%f')[:-3]
                            status_slots[camera_id].text(f"업데이트: {status_time}")
            
            # UI 업데이트 간격 (더 빠른 응답성)
            time.sleep(0.05)
    except Exception as e:
        # 오류 표시 개선
        st.error(f"오류가 발생했습니다. 페이지를 새로 고침해주세요.")

# 메인 화면 UI - 운동 선택 화면
def main_page():
    st.title("KOMI 운동 가이드")
    
    # 처음 로드 시 운동 목록 가져오기
    if not st.session_state.exercises:
        with st.spinner("운동 목록을 가져오는 중..."):
            st.session_state.exercises = get_exercises()
    
    # 운동 목록이 없으면 표시 후 종료
    if not st.session_state.exercises:
        st.info("서버에서 운동 정보를 가져올 수 없습니다.")
        if st.button("새로고침"):
            st.session_state.exercises = get_exercises()
            st.rerun()
        return
    
    # 운동 선택 칸 표시
    st.subheader("원하는 운동을 선택하세요")
    
    # 운동 선택 레이아웃
    cols = st.columns(len(st.session_state.exercises))
    
    for i, exercise in enumerate(st.session_state.exercises):
        with cols[i]:
            st.write(f"### {exercise['name']}")
            st.write(exercise['description'])
            if st.button(f"{exercise['name']} 선택", key=f"ex_{exercise['id']}"):
                st.session_state.selected_exercise = exercise
                st.session_state.page = "exercise_guide"
                st.rerun()

# 운동 가이드 화면
def exercise_guide():
    exercise = st.session_state.selected_exercise
    
    st.title(f"{exercise['name']} 가이드")
    st.write(exercise['description'])
    
    # 뒤로가기 버튼
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← 뒤로가기"):
            st.session_state.page = "main"
            st.rerun()
    
    # 운동하기 버튼
    with col3:
        if st.button("운동하기 →"):
            st.session_state.page = "exercise_webcam"
            st.rerun()
    
    # 가이드 영상 표시
    st.subheader("가이드 영상")
    
    # 영상 표시 레이아웃
    col1, col2 = st.columns(2)
    
    # 전면 영상
    with col1:
        st.write("#### 전면 영상")
        front_video_path = f"{API_URL}/data{exercise['guide_videos']['front']}"
        st.video(front_video_path)
    
    # 측면 영상
    with col2:
        st.write("#### 측면 영상")
        side_video_path = f"{API_URL}/data{exercise['guide_videos']['side']}"
        st.video(side_video_path)


def exercise_webcam():
    global selected_cameras, is_running, server_time_offset, last_time_sync
    
    exercise = st.session_state.selected_exercise
    
    st.title(f"{exercise['name']} 운동 모니터링")
    
    # 뒤로가기 버튼
    if st.button("← 가이드로 돌아가기"):
        st.session_state.page = "exercise_guide"
        st.rerun()
    
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
    
    # 카메라 자동 선택 (사용자 선택 UI 제거)
    if st.session_state.cameras:
        # 카메라가 1대인 경우 해당 카메라만 선택
        # 카메라가 2대 이상인 경우 처음 2대 선택
        auto_selected = st.session_state.cameras[:min(2, len(st.session_state.cameras))]
        
        if auto_selected != st.session_state.selected_cameras:
            st.session_state.selected_cameras = auto_selected
            selected_cameras = auto_selected
            # 카메라 선택 변경 시 동기화 버퍼 초기화
            init_sync_buffer(auto_selected)
    
    # 동기화 설정
    col1, col2 = st.columns([3, 1])
    with col1:
        # 비어있는 텍스트 표시 (동기화 상태 준비 중 메시지 제거)
        st.caption("")
    with col2:
        # 체크박스 제거하고 동기화는 항상 활성화
        use_sync = True
    
    # 카메라 슬롯 설정
    if st.session_state.selected_cameras:
        cols = st.columns(min(2, len(st.session_state.selected_cameras)))
        image_slots = {}
        status_slots = {}
        connection_indicators = {}
        
        # 카메라 개수에 따라 다르게 표시
        camera_count = len(st.session_state.selected_cameras)
        
        if camera_count == 1:
            # 카메라가 1대인 경우 정면으로 표시
            camera_id = st.session_state.selected_cameras[0]
            with cols[0]:
                header_col1, header_col2 = st.columns([4, 1])
                with header_col1:
                    # st.subheader(f"정면 카메라: {camera_id}")
                    st.subheader(f"정면")
                with header_col2:
                    connection_indicators[camera_id] = st.empty()
                
                image_slots[camera_id] = st.empty()
                status_slots[camera_id] = st.empty()
                status_slots[camera_id].text("실시간 스트리밍 준비 중...")
        else:
            # 카메라가 2대인 경우 정면/측면으로 표시
            for i, camera_id in enumerate(st.session_state.selected_cameras[:2]):
                # view_type = "정면" if i == 0 else "측면"
                with cols[i]:
                    header_col1, header_col2 = st.columns([4, 1])
                    with header_col1:
                        # st.subheader(f"{view_type} 카메라: {camera_id}")
                        st.subheader(f"측면")
                    with header_col2:
                        connection_indicators[camera_id] = st.empty()
                    
                    image_slots[camera_id] = st.empty()
                    status_slots[camera_id] = st.empty()
                    status_slots[camera_id].text("실시간 스트리밍 준비 중...")
    
    # 별도 스레드 시작 (단 한번만)
    if 'thread_started' not in st.session_state:
        thread = threading.Thread(
            target=run_async_loop,
            args=(
                API_URL, selected_cameras, is_running, sync_buffer, server_time_offset,
                last_time_sync, TIME_SYNC_INTERVAL, process_image_in_thread, thread_pool
            ),
            daemon=True
        )
        thread.start()
        st.session_state.thread_started = True
    
    # UI 업데이트 루프 실행
    update_camera_loop(image_slots, status_slots, connection_indicators, use_sync)


# 카메라 테스트 UI
def camera_test():
    global selected_cameras, is_running, server_time_offset, last_time_sync
    
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
    
    # 카메라 다중 선택기
    if st.session_state.cameras:
        # 기본값 설정 (이전에 선택된 카메라 유지)
        default_cameras = st.session_state.selected_cameras if st.session_state.selected_cameras else st.session_state.cameras[:min(2, len(st.session_state.cameras))]
        
        selected = st.multiselect(
            "모니터링할 카메라 선택 (최대 2대)",
            st.session_state.cameras,
            default=default_cameras,
            max_selections=2
        )
        
        if selected != st.session_state.selected_cameras:
            st.session_state.selected_cameras = selected
            selected_cameras = selected
            # 카메라 선택 변경 시 동기화 버퍼 초기화
            init_sync_buffer(selected)
    
    # 동기화 설정
    col1, col2 = st.columns([3, 1])
    with col1:
        # 비어있는 텍스트 표시 (동기화 상태 준비 중 메시지 제거)
        st.caption("")
    with col2:
        # 체크박스 제거하고 동기화는 항상 활성화
        use_sync = True
    
    # 두 개의 열로 이미지 배치
    if st.session_state.selected_cameras:
        cols = st.columns(min(2, len(st.session_state.selected_cameras)))
        image_slots = {}
        status_slots = {}
        connection_indicators = {}
        
        # 각 카메라별 이미지 슬롯 생성
        for i, camera_id in enumerate(st.session_state.selected_cameras[:2]):
            with cols[i]:
                header_col1, header_col2 = st.columns([4, 1])
                with header_col1:
                    st.subheader(f"카메라 {i+1}: {camera_id}")
                with header_col2:
                    # 연결 상태 표시
                    connection_indicators[camera_id] = st.empty()
                
                image_slots[camera_id] = st.empty()
                status_slots[camera_id] = st.empty()
                status_slots[camera_id].text("실시간 스트리밍 준비 중...")
    
    # 별도 스레드 시작 (단 한번만)
    if 'thread_started' not in st.session_state:
        thread = threading.Thread(
            target=run_async_loop,
            args=(
                API_URL, selected_cameras, is_running, sync_buffer, server_time_offset,
                last_time_sync, TIME_SYNC_INTERVAL, process_image_in_thread, thread_pool
            ),
            daemon=True
        )
        thread.start()
        st.session_state.thread_started = True
    
    # UI 업데이트 루프 실행
    update_camera_loop(image_slots, status_slots, connection_indicators, use_sync)
 
# 전체 앱 라우팅
def main():
    # 페이지 라우팅
    if st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "exercise_guide":
        exercise_guide()
    elif st.session_state.page == "exercise_webcam":
        exercise_webcam()

# 애플리케이션 시작
if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("앱 실행 오류가 발생했습니다. 페이지를 새로 고침해주세요.")
    finally:
        # 종료 플래그 설정
        is_running = False
        time.sleep(0.5)
        
        # 스레드 풀 종료
        thread_pool.shutdown()
