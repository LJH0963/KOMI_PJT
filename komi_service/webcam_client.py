import cv2
import asyncio
import aiohttp
import base64
import threading
import argparse
import time
import json
import os
import signal
from datetime import datetime, timedelta

# 스레드별 전용 세션과 이벤트 루프
thread_local = threading.local()

# 전역 상태 변수
running = True
cameras = {}
api_url = "http://localhost:8000"
server_time_offset = 0.0  # 서버와의 시간차 (초 단위)

# 서버 시간 동기화 함수
async def sync_server_time():
    """서버 시간과 로컬 시간의 차이를 계산"""
    global server_time_offset
    try:
        session = await init_session()
        
        # 시간 동기화를 위해 여러 번 요청하여 평균 계산
        offsets = []
        for _ in range(5):  # 5회 시도하여 평균 계산
            local_time_before = time.time()
            async with session.get(f"{api_url}/server_time", timeout=2) as response:
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

# 비동기 세션 초기화
async def init_session():
    """비동기 세션 초기화 (스레드별)"""
    if not get_session():
        thread_local.session = aiohttp.ClientSession()
    return thread_local.session

# 비동기 세션 종료
async def close_session():
    """현재 스레드의 세션 닫기"""
    session = get_session()
    if session:
        await session.close()
        thread_local.session = None

# 종료 시그널 핸들러
def handle_exit(signum, frame):
    """프로그램 종료 핸들러"""
    global running
    print("종료 요청을 받았습니다. 정리 중...")
    running = False

# 카메라 초기화
def init_camera(camera_index, resolution=(320, 240), fps=10):
    """카메라 초기화 및 설정"""
    try:
        # 카메라 객체 생성
        camera = cv2.VideoCapture(camera_index)
        
        # 해상도 설정
        if resolution:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # FPS 설정
        if fps:
            camera.set(cv2.CAP_PROP_FPS, fps)
        
        # 카메라 연결 확인
        if not camera.isOpened():
            print(f"카메라 {camera_index} 연결 실패")
            return None
            
        # 설정 확인
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        
        print(f"카메라 {camera_index} 초기화 완료:")
        print(f"  - 해상도: {actual_width}x{actual_height}")
        print(f"  - FPS: {actual_fps}")
        
        return camera
    except Exception as e:
        print(f"카메라 초기화 오류: {str(e)}")
        return None

# 이미지 인코딩
def encode_image(frame, quality=90):
    """CV2 프레임을 JPEG으로 인코딩 후 Base64 변환"""
    try:
        # JPEG 인코딩 파라미터 설정
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        # JPEG으로 압축
        result, encimg = cv2.imencode('.jpg', frame, encode_param)
        
        if not result:
            print("이미지 인코딩 실패")
            return None
            
        # Base64로 인코딩
        base64_image = base64.b64encode(encimg).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"이미지 인코딩 오류: {str(e)}")
        return None

# 비동기 서버 통신
async def register_camera(camera_id, camera_info):
    """서버에 카메라 등록"""
    try:
        session = await init_session()
        payload = {
            "camera_id": camera_id,
            "info": camera_info
        }
        
        # WebSocket URL을 HTTP URL로 변환
        http_url = api_url
        if http_url.startswith("ws://"):
            http_url = "http://" + http_url[5:].split("/")[0]
        
        async with session.post(
            f"{http_url}/register_camera", 
            json=payload, 
            timeout=5
        ) as response:
            if response.status == 200:
                print(f"카메라 {camera_id} 등록 성공")
                return True
            else:
                text = await response.text()
                print(f"카메라 등록 실패: {text}")
                return False
    except Exception as e:
        print(f"카메라 등록 오류: {str(e)}")
        return False

# 비동기 이미지 업로드
async def upload_image(camera_id, image_data, timestamp):
    """서버에 이미지 업로드"""
    try:
        if not image_data:
            return False
            
        session = await init_session()
        payload = {
            "camera_id": camera_id,
            "image_data": image_data,
            "timestamp": timestamp.isoformat()
        }
        
        # WebSocket URL을 HTTP URL로 변환
        http_url = api_url
        if http_url.startswith("ws://"):
            http_url = "http://" + http_url[5:].split("/")[0]
        
        async with session.post(
            f"{http_url}/upload_image", 
            json=payload, 
            timeout=2
        ) as response:
            if response.status != 200:
                text = await response.text()
                print(f"이미지 업로드 실패: {text}")
                return False
            return True
    except asyncio.TimeoutError:
        print(f"이미지 업로드 타임아웃")
        return False
    except Exception as e:
        print(f"이미지 업로드 오류: {str(e)}")
        return False

# 메인 카메라 처리 루프
async def camera_loop(camera_id, camera):
    """단일 카메라 처리 비동기 루프"""
    global running
    
    # 카메라 정보 구성
    camera_info = {
        "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(camera.get(cv2.CAP_PROP_FPS))
    }
    
    # 서버 시간 동기화
    await sync_server_time()
    
    # 카메라 등록
    registration_success = await register_camera(camera_id, camera_info)
    if not registration_success:
        print(f"카메라 {camera_id} 등록 실패로 종료")
        return
    
    last_upload_time = datetime.now()
    frame_interval = 1.0 / camera_info["fps"]  # 프레임 간 시간 간격
    
    # 주기적 시간 동기화 설정
    last_sync_time = time.time()
    sync_interval = 60.0  # 60초마다 시간 동기화
    
    try:
        while running:
            # 주기적으로 서버 시간 동기화
            current_time = time.time()
            if current_time - last_sync_time >= sync_interval:
                await sync_server_time()
                last_sync_time = current_time
            
            # 프레임 캡처
            ret, frame = camera.read()
            
            if not ret:
                print(f"카메라 {camera_id}로부터 프레임 읽기 실패")
                # 재연결 시도
                time.sleep(1)
                continue
            
            current_time = datetime.now()
            time_diff = (current_time - last_upload_time).total_seconds()
            
            # 지정된 FPS에 따라 이미지 업로드
            if time_diff >= frame_interval:
                # 이미지 인코딩
                image_data = encode_image(frame)
                
                if image_data:
                    # 서버 시간 기준으로 타임스탬프 생성
                    server_timestamp = get_server_time()
                    
                    # 이미지 업로드
                    upload_success = await upload_image(camera_id, image_data, server_timestamp)
                    if upload_success:
                        last_upload_time = current_time
                    
            # 적절한 딜레이
            remaining_time = frame_interval - (datetime.now() - current_time).total_seconds()
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
            else:
                await asyncio.sleep(0.001)  # 최소 딜레이
                
    except asyncio.CancelledError:
        print(f"카메라 {camera_id} 루프 취소됨")
    except Exception as e:
        print(f"카메라 {camera_id} 루프 오류: {str(e)}")
    finally:
        print(f"카메라 {camera_id} 연결 종료")
        camera.release()

# 카메라 스레드 함수
def run_camera_thread(camera_id, camera_index):
    """개별 카메라 처리 스레드"""
    try:
        # 이벤트 루프 설정
        loop = get_event_loop()
        
        # 카메라 초기화
        camera = init_camera(camera_index)
        
        if not camera:
            print(f"카메라 {camera_id} 초기화 실패")
            return
        
        # 비동기 카메라 루프 실행
        loop.run_until_complete(camera_loop(camera_id, camera))
    except Exception as e:
        print(f"카메라 스레드 오류: {str(e)}")
    finally:
        # 세션 정리
        try:
            if loop.is_running():
                loop.run_until_complete(close_session())
            else:
                asyncio.run(close_session())
        except Exception as e:
            print(f"세션 정리 오류: {str(e)}")

# 메인 함수
def main():
    """메인 프로그램"""
    global running, api_url
    
    # 인자 파서 설정
    parser = argparse.ArgumentParser(description="KOMI 웹캠 클라이언트")
    parser.add_argument('--cameras', type=str, required=True, 
                        help='카메라 설정 (형식: "카메라ID:인덱스,카메라ID:인덱스,...")')
    parser.add_argument('--server', type=str, default="http://localhost:8000",
                        help='서버 URL (기본값: http://localhost:8000)')
    parser.add_argument('--sync-time', action='store_true',
                        help='서버 시간과 동기화 활성화 (기본: 활성화)')
    
    # 인자 파싱
    args = parser.parse_args()
    api_url = args.server
    
    # 종료 시그널 핸들러 등록
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # 카메라 설정 파싱
    camera_configs = {}
    
    for cam_config in args.cameras.split(','):
        cam_parts = cam_config.strip().split(':')
        if len(cam_parts) == 2:
            camera_id, camera_index = cam_parts
            try:
                camera_configs[camera_id] = int(camera_index)
            except ValueError:
                print(f"잘못된 카메라 인덱스: {camera_index}")
    
    if not camera_configs:
        print("사용 가능한 카메라 없음")
        return
    
    print(f"서버 URL: {api_url}")
    print(f"카메라 설정: {camera_configs}")
    
    # 카메라 스레드 시작
    threads = []
    for camera_id, camera_index in camera_configs.items():
        thread = threading.Thread(
            target=run_camera_thread,
            args=(camera_id, camera_index),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        print(f"카메라 {camera_id} 스레드 시작됨")
    
    # 메인 스레드 유지
    try:
        while running and any(t.is_alive() for t in threads):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("키보드 인터럽트 감지")
        running = False
    
    # 종료 처리
    print("종료 중...")
    
    # 모든 스레드가 종료될 때까지 대기
    for thread in threads:
        thread.join(timeout=5.0)
    
    print("프로그램 종료")

if __name__ == "__main__":
    main() 