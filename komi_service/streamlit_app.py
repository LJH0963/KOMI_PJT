import streamlit as st

# 페이지 설정 (스크립트의 첫 번째 Streamlit 명령어로 이동)
st.set_page_config(page_title="KOMI 모니터링", layout="wide")

import json
import time
import numpy as np
import cv2
import base64
from datetime import datetime
from PIL import Image
from io import BytesIO
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# 서버 URL 설정
API_URL = "http://localhost:8000"  # 기본값

# Streamlit의 query parameter를 사용하여 서버 URL을 설정
if 'server_url' in st.query_params:
    API_URL = st.query_params['server_url']

# 스레드 안전 데이터 구조
image_queues = {}  # 카메라별 이미지 큐
is_running = True
selected_cameras = []

# 스레드별 전용 세션과 이벤트 루프
thread_local = threading.local()

# 이미지 처리를 위한 스레드 풀
thread_pool = ThreadPoolExecutor(max_workers=4)  # 동시에 두 카메라 처리를 위해 4개로 증가

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

# 스레드별 세션 및 이벤트 루프 관리
def get_session():
    """현재 스레드의 세션 반환 (없으면 생성)"""
    if not hasattr(thread_local, "session"):
        thread_local.session = None
    return thread_local.session

def get_event_loop():
    """현재 스레드의 이벤트 루프 반환 (없으면 생성)"""
    if not hasattr(thread_local, "loop"):
        thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_local.loop)
    return thread_local.loop

# 비동기 HTTP 클라이언트 세션 초기화
async def init_session():
    """비동기 세션 초기화 (스레드별)"""
    if not get_session():
        thread_local.session = aiohttp.ClientSession()
    return thread_local.session

# 비동기 HTTP 클라이언트 세션 종료
async def close_session():
    """현재 스레드의 세션 닫기"""
    session = get_session()
    if session:
        await session.close()
        thread_local.session = None

# 동기 함수에서 비동기 작업 실행을 위한 헬퍼 함수
def run_async(coroutine):
    """동기 함수에서 비동기 코루틴 실행"""
    loop = get_event_loop()
    return loop.run_until_complete(coroutine)

# 비동기 카메라 목록 가져오기
async def async_get_cameras():
    """비동기적으로 카메라 목록 가져오기"""
    try:
        session = await init_session()
        async with session.get(f"{API_URL}/cameras", timeout=2) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("cameras", []), "연결됨"
            return [], "오류"
    except Exception as e:
        print(f"카메라 목록 요청 오류: {str(e)}")
        return [], "연결 실패"

# 동기 카메라 목록 가져오기 (Streamlit 호환용)
def get_cameras():
    """카메라 목록 가져오기 (동기 래퍼)"""
    cameras, status = run_async(async_get_cameras())
    st.session_state.server_status = status
    return cameras

# 비동기 이미지 데이터 요청
async def async_get_raw_image(camera_id):
    """비동기적으로 이미지 바이너리 요청"""
    if not camera_id:
        return None
        
    try:
        session = await init_session()
        async with session.get(f"{API_URL}/get-image/{camera_id}", timeout=0.5) as response:
            if response.status == 200:
                return await response.read()
            return None
    except Exception as e:
        print(f"이미지 요청 오류: {str(e)}")
        return None

# 비동기 JSON 이미지 데이터 요청
async def async_get_camera_image(camera_id):
    """비동기적으로 이미지 데이터 요청"""
    if not camera_id:
        return None
        
    try:
        session = await init_session()
        async with session.get(f"{API_URL}/latest_image/{camera_id}", timeout=0.5) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("image_data")
            return None
    except Exception as e:
        print(f"이미지 요청 오류: {str(e)}")
        return None

# 스레드에서 이미지 디코딩 처리
def process_image_in_thread(image_data):
    """별도 스레드에서 이미지 처리"""
    try:
        if not image_data:
            return None
            
        if isinstance(image_data, str) and image_data.startswith('http'):
            # URL인 경우 (향후 확장성)
            return None
        else:
            # Base64 이미지인 경우
            return decode_image(image_data)
    except Exception as e:
        print(f"이미지 처리 오류: {str(e)}")
        return None

# 비동기 이미지 처리 워크플로우
async def async_image_workflow(camera_id):
    """이미지 처리 전체 워크플로우"""
    if not camera_id:
        return False
        
    try:
        # 먼저 바이너리 이미지 시도
        img_data = await async_get_raw_image(camera_id)
        if img_data:
            loop = get_event_loop()
            # 별도 스레드에서 이미지 처리
            future = loop.run_in_executor(thread_pool, BytesIO, img_data)
            img_bytes = await future
            
            try:
                image = Image.open(img_bytes)
                # 타임스탬프 추가
                timestamp = datetime.now()
                # 이미지 결과를 큐에 저장
                if camera_id not in image_queues:
                    image_queues[camera_id] = queue.Queue(maxsize=1)
                    
                if not image_queues[camera_id].full():
                    image_queues[camera_id].put({
                        "image": image,
                        "time": timestamp
                    })
                return True
            except Exception as e:
                print(f"이미지 처리 오류: {str(e)}")
        
        # 실패 시 JSON API 시도
        image_data = await async_get_camera_image(camera_id)
        if image_data:
            loop = get_event_loop()
            # 별도 스레드에서 이미지 디코딩
            future = loop.run_in_executor(thread_pool, process_image_in_thread, image_data)
            image = await future
            
            if image is not None:
                # 타임스탬프 추가
                timestamp = datetime.now()
                # 이미지 결과를 큐에 저장
                if camera_id not in image_queues:
                    image_queues[camera_id] = queue.Queue(maxsize=1)
                    
                if not image_queues[camera_id].full():
                    image_queues[camera_id].put({
                        "image": image,
                        "time": timestamp
                    })
                return True
    except Exception as e:
        print(f"워크플로우 오류: {str(e)}")
    
    return False

# 여러 카메라의 이미지를 병렬로 처리
async def process_multiple_cameras(camera_ids):
    """여러 카메라의 이미지를 병렬로 처리"""
    tasks = []
    for camera_id in camera_ids:
        if camera_id:
            tasks.append(async_image_workflow(camera_id))
    
    if tasks:
        await asyncio.gather(*tasks)

# 비동기 이미지 업데이트 함수
async def update_images():
    """백그라운드에서 이미지를 가져오는 함수"""
    global selected_cameras, is_running
    
    # 세션 초기화
    await init_session()
    
    try:
        while is_running:
            # 전역 변수로 카메라 ID 목록 확인
            camera_ids = selected_cameras
            
            if camera_ids:
                # 모든 선택된 카메라 이미지 동시에 처리
                await process_multiple_cameras(camera_ids)
            
            # 요청 간격 조절
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # 작업 취소가 요청된 경우 (정상적인 종료)
        print("이미지 업데이트 작업이 취소되었습니다.")
    except Exception as e:
        print(f"이미지 업데이트 오류: {str(e)}")
    finally:
        # 사용했던 세션 정리
        await close_session()

# 백그라운드 스레드에서 비동기 루프 실행
def run_async_loop():
    """비동기 루프를 실행하는 스레드 함수"""
    # 이 스레드 전용 이벤트 루프 생성
    loop = get_event_loop()
    
    try:
        # 이미지 업데이트 태스크 생성
        task = loop.create_task(update_images())
        
        # 이벤트 루프 실행
        loop.run_until_complete(task)
    except Exception as e:
        print(f"비동기 루프 오류: {str(e)}")
    finally:
        # 실행 중인 태스크 모두 취소
        for task in asyncio.all_tasks(loop):
            task.cancel()
        
        # 취소된 태스크 완료 대기
        if asyncio.all_tasks(loop):
            loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
        
        # 루프 종료하지 않고 닫힘 상태로만 변경
        loop.stop()

# 메인 UI
def main():
    global selected_cameras, is_running
    
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
    
    # 두 개의 열로 이미지 배치
    if st.session_state.selected_cameras:
        cols = st.columns(min(2, len(st.session_state.selected_cameras)))
        image_slots = {}
        status_slots = {}
        
        # 각 카메라별 이미지 슬롯 생성
        for i, camera_id in enumerate(st.session_state.selected_cameras[:2]):  # 최대 2개
            with cols[i]:
                st.subheader(f"카메라 {i+1}: {camera_id}")
                image_slots[camera_id] = st.empty()
                status_slots[camera_id] = st.empty()
                status_slots[camera_id].text("실시간 스트리밍 중...")
    
    # 별도 스레드 시작 (단 한번만)
    if 'thread_started' not in st.session_state:
        thread = threading.Thread(target=run_async_loop, daemon=True)
        thread.start()
        st.session_state.thread_started = True
    
    # 메인 UI 업데이트 루프
    try:
        while True:
            # 각 카메라별 이미지 큐에서 데이터 가져오기
            for camera_id in st.session_state.selected_cameras[:2]:
                if camera_id in image_queues and not image_queues[camera_id].empty():
                    img_data = image_queues[camera_id].get(block=False)
                    st.session_state.camera_images[camera_id] = img_data.get("image")
                    st.session_state.image_update_time[camera_id] = img_data.get("time")
            
            # 이미지 업데이트
            for camera_id in st.session_state.selected_cameras[:2]:
                if camera_id in st.session_state.camera_images:
                    image_slots[camera_id].image(st.session_state.camera_images[camera_id], use_container_width=True)
                    status_time = st.session_state.image_update_time[camera_id].strftime('%H:%M:%S')
                    status_slots[camera_id].text(f"업데이트: {status_time}")
            
            time.sleep(0.1)  # UI 업데이트 간격
    except Exception as e:
        st.error(f"오류: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"앱 실행 오류: {str(e)}")
    finally:
        # 종료 플래그 설정
        is_running = False
        time.sleep(0.5)  # 백그라운드 스레드가 종료 플래그를 인식할 시간 부여
        
        # 스레드 풀 종료
        thread_pool.shutdown() 