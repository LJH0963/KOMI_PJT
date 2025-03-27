import asyncio
import aiohttp
import collections
import queue
import random
import time
from datetime import datetime

from frontend import config
from frontend import utils
from frontend.session import app_state

# 동기화 버퍼 초기화
def init_sync_buffer(camera_ids):
    """동기화 버퍼 초기화 함수"""
    for camera_id in camera_ids:
        if camera_id not in app_state.sync_buffer:
            app_state.sync_buffer[camera_id] = collections.deque(maxlen=config.SYNC_BUFFER_SIZE)

# 동기화된 프레임 쌍 찾기
def find_synchronized_frames(selected_cameras):
    """여러 카메라에서 타임스탬프가 가장 가까운 프레임 쌍 찾기"""
    if len(selected_cameras) < 2:
        return None, None
    
    if not all(camera_id in app_state.sync_buffer and len(app_state.sync_buffer[camera_id]) > 0 
               for camera_id in selected_cameras):
        return None, None
    
    # 가장 최근의 타임스탬프 기준으로 시작
    latest_frame_times = {
        camera_id: max([frame["time"] for frame in app_state.sync_buffer[camera_id]])
        for camera_id in selected_cameras
    }
    
    # 가장 늦은 타임스탬프 찾기
    reference_time = min(latest_frame_times.values())
    
    # 각 카메라에서 기준 시간과 가장 가까운 프레임 찾기
    best_frames = {}
    for camera_id in selected_cameras:
        closest_frame = min(
            [frame for frame in app_state.sync_buffer[camera_id]],
            key=lambda x: abs((x["time"] - reference_time).total_seconds())
        )
        
        # 시간 차이가 허용 범위 내인지 확인
        time_diff_ms = abs((closest_frame["time"] - reference_time).total_seconds() * 1000)
        if time_diff_ms <= config.MAX_SYNC_DIFF_MS:
            best_frames[camera_id] = closest_frame
        else:
            return None, None
    
    # 모든 카메라에 대해 프레임을 찾았는지 확인
    if len(best_frames) == len(selected_cameras):
        # 동기화 상태 업데이트
        max_diff = max([abs((f["time"] - reference_time).total_seconds() * 1000) 
                        for f in best_frames.values()])
        
        # 동기화 상태 메시지 반환
        sync_status = f"동기화됨 (최대 차이: {max_diff:.1f}ms)"
        return best_frames, sync_status
    
    return None, None

# 연결 상태 업데이트 함수
def update_connection_status(camera_id, status):
    """카메라 연결 상태 업데이트"""
    # 전역 상태 업데이트
    app_state.ws_connection_status[camera_id] = status
    
    # 연결 시도 횟수 관리
    if status == "connected":
        app_state.connection_attempts[camera_id] = 0
    elif status == "disconnected":
        if camera_id not in app_state.connection_attempts:
            app_state.connection_attempts[camera_id] = 0

# 비동기 카메라 목록 가져오기
async def async_get_cameras():
    """비동기적으로 카메라 목록 가져오기"""
    try:
        session = await utils.init_session()
        # 타임아웃 파라미터를 숫자 대신 ClientTimeout 객체로 변경
        request_timeout = aiohttp.ClientTimeout(total=2)
        async with session.get(f"{config.API_URL}/cameras", timeout=request_timeout) as response:
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
    cameras, status = utils.run_async(async_get_cameras())
    return cameras, status

# 운동 목록 가져오기
async def get_exercises():
    """운동 목록을 서버에서 가져오기"""
    try:
        session = await utils.init_session()
        async with session.get(f"{config.API_URL}/exercises") as response:
            if response.status == 200:
                return await response.json()
            return None
    except Exception:
        return None

# 운동 세션 시작 함수
async def start_exercise(exercise_id, camera_ids):
    """운동 세션 시작"""
    try:
        session = await utils.init_session()
        async with session.post(
            f"{config.API_URL}/exercise/start",
            json={"exercise_id": exercise_id, "camera_ids": camera_ids}
        ) as response:
            if response.status == 200:
                return await response.json()
            return None
    except Exception:
        return None

# 운동 세션 상태 확인 함수
async def check_exercise_status(session_id):
    """운동 세션 상태 확인"""
    try:
        session = await utils.init_session()
        async with session.get(f"{config.API_URL}/exercise/{session_id}") as response:
            if response.status == 200:
                return await response.json()
            return None
    except Exception:
        return None

# 운동 세션 종료 함수
async def end_exercise(session_id):
    """운동 세션 종료"""
    try:
        session = await utils.init_session()
        async with session.post(f"{config.API_URL}/exercise/{session_id}/end") as response:
            if response.status == 200:
                return await response.json()
            return None
    except Exception:
        return None

# 서버 상태 확인 함수
async def check_server_health():
    """서버 상태 확인"""
    try:
        session = await utils.init_session()
        async with session.get(f"{config.API_URL}/health") as response:
            if response.status == 200:
                return await response.json()
            return {"status": "error", "message": f"상태 코드: {response.status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)} 