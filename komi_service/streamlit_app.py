import streamlit as st
import asyncio
import aiohttp
import threading
import json
import base64
import time
from datetime import datetime
from PIL import Image
import io
import concurrent.futures
import queue
import numpy as np

# 전역 변수 설정
API_URL = "http://localhost:8000"
camera_ids = ["front", "side"]  # 전면 카메라와 측면 카메라

# 스레드 로컬 스토리지
thread_local = threading.local()
connection_status = {}
connection_attempts = {}
image_queues = {}
latest_images = {}
latest_pose_data = {}
is_running = True

# 스레드 풀 초기화
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

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

async def init_session():
    """비동기 세션 초기화 (스레드별)"""
    if not get_session():
        # 향상된 타임아웃 설정으로 세션 생성
        timeout = aiohttp.ClientTimeout(total=30, connect=5, sock_connect=5, sock_read=5)
        thread_local.session = aiohttp.ClientSession(timeout=timeout)
    return thread_local.session

async def close_session():
    """현재 스레드의 세션 닫기"""
    session = get_session()
    if session:
        await session.close()
        thread_local.session = None

def update_connection_status(camera_id, status):
    """카메라 연결 상태 업데이트"""
    connection_status[camera_id] = status

# 비동기 HTTP 요청 함수
async def fetch_data(url, method="GET", data=None):
    """HTTP 요청 수행"""
    session = await init_session()
    try:
        if method == "GET":
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        elif method == "POST":
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
    except Exception as e:
        st.error(f"데이터 요청 실패: {str(e)}")
    return None

# 운동 목록 가져오기
def get_exercises():
    """FastAPI 서버에서 운동 목록 가져오기"""
    loop = get_event_loop()
    return loop.run_until_complete(fetch_data(f"{API_URL}/exercises"))

# 웹소켓 연결 및 이미지 수신 함수
async def connect_to_camera_stream(camera_id):
    """WebSocket을 통해 카메라 스트림에 연결"""
    global latest_images, latest_pose_data
    
    # 카메라 별 큐 초기화
    if camera_id not in image_queues:
        image_queues[camera_id] = queue.Queue(maxsize=10)
    
    # 연결 상태 초기화
    if camera_id not in connection_status:
        connection_status[camera_id] = "disconnected"
    
    # 연결 시도 횟수 초기화
    if camera_id not in connection_attempts:
        connection_attempts[camera_id] = 0
    
    # 이미 연결된 경우 리턴
    if connection_status.get(camera_id) == "connecting":
        return
    
    # 연결 상태 업데이트
    update_connection_status(camera_id, "connecting")
    
    try:
        session = await init_session()
        # WebSocket URL 구성
        ws_url = f"{API_URL.replace('http://', 'ws://')}/ws/stream/{camera_id}"
        
        # 향상된 WebSocket 옵션
        heartbeat = 30.0  # 30초 핑/퐁
        ws_timeout = aiohttp.ClientWSTimeout(ws_close=60.0)  # WebSocket 종료 대기 시간 60초
        
        async with session.ws_connect(
            ws_url, 
            heartbeat=heartbeat,
            timeout=ws_timeout,
            max_msg_size=0,  # 무제한
            compress=False  # 웹소켓 압축 비활성화로 성능 향상
        ) as ws:
            # 연결 성공 - 상태 업데이트 및 시도 횟수 초기화
            update_connection_status(camera_id, "connected")
            connection_attempts[camera_id] = 0
            
            last_ping_time = time.time()
            ping_interval = 25
            
            while is_running:
                # 핑 전송 (주기적으로) - 서버 핑/퐁 메커니즘과 별개로 유지
                current_time = time.time()
                if current_time - last_ping_time >= ping_interval:
                    try:
                        await ws.ping()
                        last_ping_time = current_time
                    except:
                        # 핑 실패 시 루프 탈출하여 재연결
                        break
                
                # 데이터 수신 (짧은 타임아웃으로 반응성 유지)
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
                    
                    if msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            
                            if data.get("type") in ["image", "image_with_pose"]:
                                image_data = data.get("image_data")
                                timestamp = data.get("timestamp")
                                pose_data = data.get("pose_data")
                                
                                # 이미지를 큐에 추가하고 메타데이터 저장
                                if image_data:
                                    # 최신 이미지 저장
                                    latest_images[camera_id] = {
                                        "image_data": image_data,
                                        "timestamp": timestamp
                                    }
                                    
                                    # 포즈 데이터가 있으면 저장
                                    if pose_data:
                                        latest_pose_data[camera_id] = pose_data
                                    
                                    # 큐에 데이터 추가 (가득 차면 가장 오래된 항목 삭제)
                                    try:
                                        # 큐가 가득 찼으면 이전 항목 제거
                                        if image_queues[camera_id].full():
                                            image_queues[camera_id].get_nowait()
                                        
                                        # 새 이미지 큐에 추가
                                        image_queues[camera_id].put_nowait({
                                            "image_data": image_data,
                                            "timestamp": timestamp,
                                            "pose_data": pose_data if pose_data else None
                                        })
                                    except queue.Full:
                                        pass  # 큐가 꽉 찬 경우 무시
                        except json.JSONDecodeError:
                            pass
                            
                except asyncio.TimeoutError:
                    # 타임아웃은 정상, 계속 진행
                    pass
                
    except Exception as e:
        st.error(f"카메라 {camera_id} 연결 오류: {str(e)}")
    finally:
        update_connection_status(camera_id, "disconnected")

# 웹소켓 연결 관리 스레드 함수
def camera_stream_thread(camera_id):
    """비동기 이벤트 루프를 실행하는 스레드 함수"""
    loop = get_event_loop()
    
    try:
        # 비동기 연결 함수 실행
        loop.run_until_complete(connect_to_camera_stream(camera_id))
    except Exception as e:
        st.error(f"카메라 스트림 스레드 오류: {str(e)}")
    finally:
        # 세션 닫기
        try:
            loop.run_until_complete(close_session())
        except:
            pass

# 스트리밍 시작 함수
def start_streaming():
    """모든 카메라에 대한 스트리밍 시작"""
    global is_running
    is_running = True
    
    # 카메라별 이미지 큐 초기화
    for camera_id in camera_ids:
        if camera_id not in image_queues:
            image_queues[camera_id] = queue.Queue(maxsize=10)
    
    # 각 카메라별 스트리밍 스레드 시작
    for camera_id in camera_ids:
        thread_pool.submit(camera_stream_thread, camera_id)

# 스트리밍 중지 함수
def stop_streaming():
    """모든 카메라 스트리밍 중지"""
    global is_running
    is_running = False
    
    # 큐 정리
    for camera_id in camera_ids:
        if camera_id in image_queues:
            try:
                while not image_queues[camera_id].empty():
                    image_queues[camera_id].get_nowait()
            except:
                pass

# 이미지 디코딩 및 표시 함수
def display_image(image_data, key, width=None):
    """Base64 인코딩된 이미지 데이터를 디코딩하여 표시"""
    try:
        if image_data:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if width:
                st.image(image, width=width, key=key)
            else:
                st.image(image, key=key)
            return True
    except Exception as e:
        st.error(f"이미지 디코딩 오류: {str(e)}")
    return False

# 카메라 상태 제어 함수
def control_camera_status(camera_id, status):
    """카메라 상태 제어 API 호출"""
    loop = get_event_loop()
    result = loop.run_until_complete(
        fetch_data(
            f"{API_URL}/cameras/{camera_id}/status",
            method="POST",
            data={"status": status}
        )
    )
    return result

# 페이지 관리 함수
def set_page(page_name, **kwargs):
    """페이지 상태 설정 및 저장"""
    st.session_state.page = page_name
    # 추가 인자가 있으면 세션 상태에 저장
    for key, value in kwargs.items():
        st.session_state[key] = value

# 메인 페이지 (운동 선택)
def main_page():
    """메인 페이지 - 운동 선택 화면"""
    st.title("KOMI 운동 보조 시스템")
    
    # 운동 목록 가져오기
    exercise_data = get_exercises()
    
    if not exercise_data or "exercises" not in exercise_data:
        st.error("운동 데이터를 가져오는데 실패했습니다.")
        return
    
    # 운동 선택 화면 구성
    st.subheader("운동을 선택하세요")
    
    # 그리드 레이아웃 시작
    cols = st.columns(3)
    
    # 각 운동을 카드 형태로 표시
    for i, exercise in enumerate(exercise_data["exercises"]):
        with cols[i % 3]:
            st.subheader(exercise["name"])
            st.text(exercise["description"])
            
            # 버튼 클릭 시 운동 가이드 페이지로 이동
            if st.button(f"{exercise['name']} 선택", key=f"select_{exercise['id']}"):
                set_page("exercise_guide", exercise_id=exercise["id"])

# 운동 가이드 페이지
def exercise_guide():
    """운동 가이드 페이지 - 선택한 운동의 가이드 영상 표시"""
    # 선택된 운동 ID 확인
    if "exercise_id" not in st.session_state:
        st.error("선택된 운동이 없습니다.")
        if st.button("운동 선택으로 돌아가기"):
            set_page("main_page")
        return
    
    exercise_id = st.session_state.exercise_id
    
    # 운동 상세 정보 가져오기
    loop = get_event_loop()
    exercise = loop.run_until_complete(fetch_data(f"{API_URL}/exercise/{exercise_id}"))
    
    if not exercise:
        st.error("운동 정보를 가져오는데 실패했습니다.")
        if st.button("운동 선택으로 돌아가기"):
            set_page("main_page")
        return
    
    # 헤더 표시
    st.title(f"{exercise['name']} 가이드")
    st.text(exercise["description"])
    
    # 가이드 영상 표시
    if "guide_videos" in exercise:
        st.subheader("가이드 영상")
        
        # 2열 레이아웃
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**전면 영상**")
            if "front" in exercise["guide_videos"]:
                front_video = exercise["guide_videos"]["front"]
                st.video(f"{API_URL}/data{front_video}")
        
        with col2:
            st.markdown("**측면 영상**")
            if "side" in exercise["guide_videos"]:
                side_video = exercise["guide_videos"]["side"]
                st.video(f"{API_URL}/data{side_video}")
    
    # 네비게이션 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("운동 선택으로 돌아가기"):
            set_page("main_page")
    with col2:
        if st.button("운동 시작하기"):
            set_page("exercise_webcam", exercise_id=exercise_id)

# 웹캠 모니터링 페이지
def exercise_webcam():
    """웹캠 모니터링 페이지 - 실시간 웹캠 피드백"""
    # 선택된 운동 ID 확인
    if "exercise_id" not in st.session_state:
        st.error("선택된 운동이 없습니다.")
        if st.button("운동 선택으로 돌아가기"):
            set_page("main_page")
        return
    
    exercise_id = st.session_state.exercise_id
    
    # 운동 상세 정보 가져오기
    loop = get_event_loop()
    exercise = loop.run_until_complete(fetch_data(f"{API_URL}/exercise/{exercise_id}"))
    
    if not exercise:
        st.error("운동 정보를 가져오는데 실패했습니다.")
        if st.button("운동 선택으로 돌아가기"):
            set_page("main_page")
        return
    
    # 헤더 표시
    st.title(f"{exercise['name']} 실시간 분석")
    
    # 뒤로가기 버튼
    if st.button("가이드로 돌아가기"):
        stop_streaming()
        set_page("exercise_guide", exercise_id=exercise_id)
    
    # 스트리밍 시작 (처음 페이지 로드 시)
    if "streaming_started" not in st.session_state:
        start_streaming()
        st.session_state.streaming_started = True
        
        # 카메라 상태 초기화
        for camera_id in camera_ids:
            # 카메라 상태를 'on'으로 설정
            control_camera_status(camera_id, "on")
    
    # 카메라 컨트롤
    st.subheader("카메라 제어")
    col1, col2 = st.columns(2)
    
    with col1:
        # 전면 카메라 상태 표시 및 제어
        front_status = connection_status.get("front", "disconnected")
        st.markdown(f"**전면 카메라**: {front_status}")
        
        # 상태에 따른 버튼 표시
        if front_status == "connected":
            if st.button("포즈 감지 시작", key="front_detect"):
                control_camera_status("front", "detect")
            if st.button("일반 모드로 전환", key="front_normal"):
                control_camera_status("front", "on")
    
    with col2:
        # 측면 카메라 상태 표시 및 제어
        side_status = connection_status.get("side", "disconnected")
        st.markdown(f"**측면 카메라**: {side_status}")
        
        # 상태에 따른 버튼 표시
        if side_status == "connected":
            if st.button("포즈 감지 시작", key="side_detect"):
                control_camera_status("side", "detect")
            if st.button("일반 모드로 전환", key="side_normal"):
                control_camera_status("side", "on")
    
    # 이미지 스트리밍 영역
    st.subheader("실시간 영상")
    
    # 이미지 슬롯 생성 (10fps * 2 카메라)
    cam_cols = st.columns(2)
    
    # 각 카메라별 이미지 슬롯을 미리 생성해두기
    with cam_cols[0]:
        st.markdown("**전면 카메라**")
        front_image_slot = st.empty()
    
    with cam_cols[1]:
        st.markdown("**측면 카메라**")
        side_image_slot = st.empty()
    
    # 포즈 데이터 디버그 영역
    st.subheader("포즈 데이터 (디버그)")
    debug_slots = {}
    for camera_id in camera_ids:
        debug_slots[camera_id] = st.empty()
    
    # Streamlit 이미지 업데이트 함수
    def update_streamlit_images():
        """이미지 슬롯에 최신 이미지 표시"""
        # 전면 카메라 이미지 표시
        if "front" in latest_images and latest_images["front"].get("image_data"):
            img_data = latest_images["front"]["image_data"]
            image_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(image_bytes))
            front_image_slot.image(image, use_column_width=True)
        
        # 측면 카메라 이미지 표시
        if "side" in latest_images and latest_images["side"].get("image_data"):
            img_data = latest_images["side"]["image_data"]
            image_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(image_bytes))
            side_image_slot.image(image, use_column_width=True)
        
        # 포즈 데이터 디버그 정보 표시
        for camera_id in camera_ids:
            if camera_id in latest_pose_data:
                debug_slots[camera_id].json(latest_pose_data[camera_id])
    
    # 페이지 초기화 시 이미지 업데이트 스레드 시작
    if "update_thread_started" not in st.session_state:
        st.session_state.update_thread_started = True
        
        # Streamlit의 experimental_rerun 대신 주기적으로 업데이트 실행
        # 이 예제에서는 update_streamlit_images()를 호출하기 위해 
        # 버튼을 숨겨두고 JavaScript로 주기적으로 클릭
        st.markdown("""
        <script>
            function clickUpdateButton() {
                const btn = document.getElementById('update_images_btn');
                if (btn) {
                    btn.click();
                }
                setTimeout(clickUpdateButton, 100); // 10fps = 100ms
            }
            setTimeout(clickUpdateButton, 500); // 페이지 로드 후 0.5초 뒤 시작
        </script>
        """, unsafe_allow_html=True)
        
        # JavaScript에서 클릭할 숨겨진 버튼
        if st.button("Update Images", key="update_images_btn", help="JavaScript에서 자동으로 호출됨"):
            update_streamlit_images()

# 앱 시작점
def main():
    """애플리케이션 메인 함수"""
    # 세션 상태 초기화
    if "page" not in st.session_state:
        st.session_state.page = "main_page"
    
    # 페이지 라우팅
    if st.session_state.page == "main_page":
        main_page()
    elif st.session_state.page == "exercise_guide":
        exercise_guide()
    elif st.session_state.page == "exercise_webcam":
        exercise_webcam()

if __name__ == "__main__":
    main() 