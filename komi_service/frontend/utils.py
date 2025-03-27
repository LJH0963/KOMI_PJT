import numpy as np
import cv2
import base64
from datetime import datetime, timedelta
import time
import json
import asyncio
import aiohttp
import threading
import random
from concurrent.futures import ThreadPoolExecutor
from frontend import config

# 스레드 안전 데이터 구조
thread_local = threading.local()
thread_pool = ThreadPoolExecutor(max_workers=4)

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
        if isinstance(image_data, str) and image_data.startswith('http'):  # URL인 경우 (향후 확장성)
            return None
        else:
            return decode_image(image_data)  # Base64 이미지인 경우
    except Exception as e:
        print(f"이미지 처리 오류: {str(e)}")
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
        # 타임아웃 설정 추가
        timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_connect=5, sock_read=5)
        thread_local.session = aiohttp.ClientSession(timeout=timeout)
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

# 포즈 데이터를 이미지에 그리는 함수
def draw_pose_on_image(image, pose_data):
    """포즈 데이터를 이미지에 시각화"""
    if image is None or pose_data is None:
        return image
    
    try:
        # 이미지 복사
        img_copy = image.copy()
        
        # 키포인트 그리기
        keypoints = pose_data.get("keypoints", [])
        if not keypoints or len(keypoints) == 0:
            return image
        
        # 첫 번째 사람의 키포인트만 사용 (여러 명이 감지된 경우)
        person_keypoints = keypoints[0]
        
        # 색상 정의 (RGB)
        keypoint_color = (255, 0, 0)  # 빨간색
        skeleton_color = (0, 255, 0)  # 녹색
        
        # 각 키포인트 그리기
        for kp in person_keypoints:
            if kp.get("x") is not None and kp.get("y") is not None:
                if kp.get("confidence", 0) > 0.5:  # 높은 신뢰도의 키포인트만
                    x, y = int(kp["x"]), int(kp["y"])
                    cv2.circle(img_copy, (x, y), 5, keypoint_color, -1)
        
        # COCO 데이터셋 기준 스켈레톤 연결 정의
        skeleton = [
            (5, 7), (7, 9), (6, 8), (8, 10),  # 팔 (좌우)
            (11, 13), (13, 15), (12, 14), (14, 16),  # 다리 (좌우)
            (5, 6), (11, 12), (5, 11), (6, 12)  # 몸통
        ]
        
        coco_keypoints = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 키포인트 부위별 딕셔너리 생성
        kp_dict = {kp["part"]: kp for kp in person_keypoints}
        
        # 스켈레톤 그리기
        for joint1_idx, joint2_idx in skeleton:
            joint1_name = coco_keypoints[joint1_idx]
            joint2_name = coco_keypoints[joint2_idx]
            
            joint1 = kp_dict.get(joint1_name, {})
            joint2 = kp_dict.get(joint2_name, {})
            
            if (joint1.get("x") is not None and joint1.get("y") is not None and 
                joint2.get("x") is not None and joint2.get("y") is not None and
                joint1.get("confidence", 0) > 0.5 and joint2.get("confidence", 0) > 0.5):
                cv2.line(img_copy, 
                        (int(joint1["x"]), int(joint1["y"])), 
                        (int(joint2["x"]), int(joint2["y"])), 
                        skeleton_color, 2)
        
        return img_copy
    except Exception as e:
        print(f"포즈 그리기 오류: {str(e)}")
        return image

# 서버 시간 동기화 함수 - 간소화 및 안정성 향상
async def sync_server_time(server_time_offset, last_time_sync):
    """서버 시간과 로컬 시간의 차이를 계산"""
    # 이미 최근에 동기화했다면 스킵
    current_time = time.time()
    if current_time - last_time_sync < config.TIME_SYNC_INTERVAL:
        return server_time_offset, last_time_sync
    
    try:
        session = await init_session()
        
        # 무작위 지연 추가 (서버 부하 분산)
        jitter = random.uniform(0, 1.0)
        await asyncio.sleep(jitter)
        
        local_time_before = time.time()
        # 타임아웃 파라미터를 ClientTimeout 객체로 변경
        request_timeout = aiohttp.ClientTimeout(total=2)
        async with session.get(f"{config.API_URL}/server_time", timeout=request_timeout) as response:
            if response.status != 200:
                return server_time_offset, last_time_sync
                
            local_time_after = time.time()
            data = await response.json()
            
            server_timestamp = data.get("timestamp")
            if not server_timestamp:
                return server_time_offset, last_time_sync
            
            network_delay = (local_time_after - local_time_before) / 2
            local_time_avg = local_time_before + network_delay
            new_server_time_offset = server_timestamp - local_time_avg
            new_last_time_sync = time.time()
            return new_server_time_offset, new_last_time_sync
    except asyncio.TimeoutError:
        # 타임아웃은 조용히 처리
        return server_time_offset, last_time_sync
    except Exception:
        # 그 외 오류도 조용히 처리
        return server_time_offset, last_time_sync

# 서버 시간 기준 현재 시간 반환
def get_server_time(server_time_offset):
    """서버 시간 기준의 현재 시간 계산"""
    return datetime.now() + timedelta(seconds=server_time_offset) 