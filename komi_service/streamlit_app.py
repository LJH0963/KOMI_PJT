import streamlit as st

# 페이지 설정 (스크립트의 첫 번째 Streamlit 명령어로 이동)
st.set_page_config(page_title="KOMI 모니터링", layout="wide")

import json
import time
import numpy as np
import cv2
import base64
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import collections

# 서버 URL 설정
API_URL = "http://localhost:8000"  # 기본값

# Streamlit의 query parameter를 사용하여 서버 URL을 설정
if 'server_url' in st.query_params:
    API_URL = st.query_params['server_url']

# 스레드 안전 데이터 구조
image_queues = {}  # 카메라별 이미지 큐
is_running = True
selected_cameras = []

# 서버 시간 동기화 관련 변수
server_time_offset = 0.0  # 서버와의 시간차 (초 단위)
last_time_sync = 0  # 마지막 시간 동기화 시간
TIME_SYNC_INTERVAL = 60  # 시간 동기화 주기 (초)

# 동기화 버퍼 설정
sync_buffer = {}  # 카메라 ID -> 최근 프레임 버퍼 (collections.deque)
SYNC_BUFFER_SIZE = 10  # 각 카메라별 버퍼 크기
MAX_SYNC_DIFF_MS = 100  # 프레임 간 최대 허용 시간 차이 (밀리초)

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
if 'sync_status' not in st.session_state:
    st.session_state.sync_status = "준비 중..."

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
    
    # 선택된 카메라가 2개 미만이면 동기화 필요 없음
    if len(st.session_state.selected_cameras) < 2:
        return None
    
    # 모든 카메라의 버퍼에 프레임이 있는지 확인
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
            # 시간 차이가 너무 크면 동기화 실패
            return None
    
    # 모든 카메라에 대해 프레임을 찾았는지 확인
    if len(best_frames) == len(st.session_state.selected_cameras):
        # 동기화 상태 업데이트
        max_diff = max([abs((f["time"] - reference_time).total_seconds() * 1000) 
                        for f in best_frames.values()])
        st.session_state.sync_status = f"동기화됨 (최대 차이: {max_diff:.1f}ms)"
        return best_frames
    
    return None

# 서버 시간 동기화 함수
async def sync_server_time():
    """서버 시간과 로컬 시간의 차이를 계산"""
    global server_time_offset, last_time_sync
    
    try:
        session = await init_session()
        
        # 시간 동기화를 위해 여러 번 요청하여 평균 계산
        offsets = []
        for _ in range(5):  # 5회 시도하여 평균 계산
            local_time_before = time.time()
            async with session.get(f"{API_URL}/server_time", timeout=2) as response:
                if response.status != 200:
                    continue
                    
                local_time_after = time.time()
                data = await response.json()
                
                # 서버 시간 파싱
                server_timestamp = data.get("timestamp")
                if not server_timestamp:
                    continue
                
                # 네트워크 지연 시간 추정 (왕복 시간의 절반)
                network_delay = (local_time_after - local_time_before) / 2
                
                # 보정된 서버 시간과 로컬 시간의 차이 계산
                local_time_avg = local_time_before + network_delay
                offset = server_timestamp - local_time_avg
                
                offsets.append(offset)
                
                # 반복 사이에 짧은 대기
                await asyncio.sleep(0.1)
                
        if offsets:
            # 이상치 제거 (최대, 최소값 제외)
            if len(offsets) > 2:
                offsets.remove(max(offsets))
                offsets.remove(min(offsets))
                
            # 평균 오프셋 계산
            server_time_offset = sum(offsets) / len(offsets)
            print(f"서버 시간 동기화 완료: 오프셋 {server_time_offset:.3f}초")
            last_time_sync = time.time()
            return True
        else:
            print("서버 시간 동기화 실패: 유효한 응답 없음")
            return False
            
    except Exception as e:
        print(f"서버 시간 동기화 오류: {str(e)}")
        return False

# 서버 시간 기준 현재 시간 반환
def get_server_time():
    """서버 시간 기준의 현재 시간 계산"""
    return datetime.now() + timedelta(seconds=server_time_offset)

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
                # 서버 기준 타임스탬프 추가
                timestamp = get_server_time()
                
                # 동기화 버퍼에 프레임 저장
                frame_data = {
                    "image": image,
                    "time": timestamp
                }
                
                if camera_id in sync_buffer:
                    sync_buffer[camera_id].append(frame_data)
                
                # 이전 방식과의 호환성을 위해 이미지 큐에도 저장
                if camera_id not in image_queues:
                    image_queues[camera_id] = queue.Queue(maxsize=1)
                    
                if not image_queues[camera_id].full():
                    image_queues[camera_id].put(frame_data)
                
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
                # 서버 기준 타임스탬프 추가
                timestamp = get_server_time()
                
                # 동기화 버퍼에 프레임 저장
                frame_data = {
                    "image": image,
                    "time": timestamp
                }
                
                if camera_id in sync_buffer:
                    sync_buffer[camera_id].append(frame_data)
                
                # 이전 방식과의 호환성을 위해 이미지 큐에도 저장
                if camera_id not in image_queues:
                    image_queues[camera_id] = queue.Queue(maxsize=1)
                    
                if not image_queues[camera_id].full():
                    image_queues[camera_id].put(frame_data)
                    
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
    global selected_cameras, is_running, last_time_sync
    
    # 세션 초기화
    await init_session()
    
    # 초기 서버 시간 동기화
    await sync_server_time()
    
    try:
        while is_running:
            # 주기적 서버 시간 동기화
            current_time = time.time()
            if current_time - last_time_sync >= TIME_SYNC_INTERVAL:
                await sync_server_time()
            
            # 전역 변수로 카메라 ID 목록 확인
            camera_ids = selected_cameras
            
            if camera_ids:
                # 동기화 버퍼 초기화
                init_sync_buffer(camera_ids)
                
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
    
    # 서버 정보 표시
    st.caption(f"서버 연결: {API_URL} (시간 오프셋: {server_time_offset:.3f}초)")
    
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
        st.caption(f"동기화 상태: {st.session_state.sync_status}")
    with col2:
        use_sync = st.checkbox("동기화 활성화", value=True)
    
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
                        image_slots[camera_id].image(st.session_state.camera_images[camera_id], use_container_width=True)
                        status_time = st.session_state.image_update_time[camera_id].strftime('%H:%M:%S.%f')[:-3]
                        status_slots[camera_id].text(f"업데이트: {status_time}")
            
            time.sleep(0.05)  # UI 업데이트 간격 (더 빠른 응답성)
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