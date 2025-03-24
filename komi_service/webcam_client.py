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
ws_connections = {}  # 카메라 ID -> WebSocket 연결

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

# 종료 시그널 핸들러
def handle_exit(signum, frame):
    """프로그램 종료 핸들러"""
    global running
    print("종료 요청을 받았습니다. 정리 중...")
    running = False

# 카메라 초기화
def init_camera(
    camera_index,
    # resolution=(640, 480),
    fps=15
):
    """카메라 초기화 및 설정"""
    try:
        # 카메라 객체 생성
        camera = cv2.VideoCapture(camera_index)
        
        # # 해상도 설정
        # if resolution:
        #     camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        #     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
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
        # print(f"  - 요청 해상도: {resolution[0]}x{resolution[1]}")
        # print(f"  - 실제 해상도: {actual_width}x{actual_height}")
        print(f"  - 요청 FPS: {fps}")
        print(f"  - 실제 FPS: {actual_fps}")
        
        return camera
    except Exception as e:
        print(f"카메라 초기화 오류: {str(e)}")
        return None

# 이미지 인코딩 함수
def encode_image(frame, quality=85, max_width=640, verbose=False):
    """이미지를 Base64 인코딩"""
    try:
        # 프레임 크기 조정 (최대 너비 초과 시)
        h, w = frame.shape[:2]
        if w > max_width:
            # 종횡비 유지하며 리사이즈
            aspect_ratio = h / w
            new_w = max_width
            new_h = int(new_w * aspect_ratio)
            
            # 빠른 리사이징을 위해 INTER_AREA 사용
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if verbose:
                print(f"이미지 리사이즈: {w}x{h} -> {new_w}x{new_h}")
        
        # 압축 전 이미지 크기 측정
        original_size = frame.size * frame.itemsize  # 바이트 단위
        
        # JPEG으로 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        
        # 압축 후 이미지 크기 측정
        compressed_size = len(encimg)
        compression_ratio = 100 - (compressed_size / original_size * 100)
        
        if verbose:
            print(f"압축률: {compression_ratio:.1f}% ({original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB)")
            
        # Base64로 인코딩
        base64_image = base64.b64encode(encimg).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"이미지 인코딩 오류: {str(e)}")
        return None

# WebSocket 연결 및 카메라 등록
async def connect_camera_websocket(camera_id, camera_info):
    """WebSocket을 통해 카메라 연결 및 등록"""
    global ws_connections
    
    try:
        session = await init_session()
        
        # WebSocket URL 생성
        if api_url.startswith('http://'):
            ws_url = f"ws://{api_url[7:]}/ws/camera"
        elif api_url.startswith('https://'):
            ws_url = f"wss://{api_url[8:]}/ws/camera"
        else:
            ws_url = f"ws://{api_url}/ws/camera"
        
        # WebSocket 연결
        ws = await session.ws_connect(ws_url, timeout=10)
        
        # 카메라 등록 메시지 전송
        register_msg = {
            "type": "register",
            "camera_id": camera_id,
            "info": camera_info
        }
        
        await ws.send_json(register_msg)
        
        # 응답 대기
        response = await ws.receive_json()
        
        if response.get("type") == "connection_successful":
            print(f"카메라 {camera_id} WebSocket 연결 및 등록 성공")
            ws_connections[camera_id] = ws
            return True
        else:
            print(f"카메라 등록 실패: {response}")
            await ws.close()
            return False
            
    except Exception as e:
        print(f"WebSocket 연결 오류: {str(e)}")
        return False

# WebSocket을 통한 이미지 전송
async def send_frame_via_websocket(camera_id, image_data, timestamp):
    """WebSocket을 통해 이미지 프레임 전송"""
    try:
        if camera_id not in ws_connections:
            print(f"카메라 {camera_id}의 WebSocket 연결이 없습니다")
            return False
            
        ws = ws_connections[camera_id]
        
        # 프레임 메시지 생성
        frame_msg = {
            "type": "frame",
            "camera_id": camera_id,
            "image_data": image_data,
            "timestamp": timestamp.isoformat()
        }
        
        # 메시지 전송
        await ws.send_json(frame_msg)
        return True
            
    except Exception as e:
        print(f"프레임 전송 오류: {str(e)}")
        
        # 연결 오류 시 재연결 시도를 위해 연결 제거
        if camera_id in ws_connections:
            del ws_connections[camera_id]
            
        return False

# 메인 카메라 처리 루프
async def camera_loop(camera_id, camera, quality=85, max_width=640):
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
    
    # WebSocket 연결 및 카메라 등록
    registration_success = await connect_camera_websocket(camera_id, camera_info)
    if not registration_success:
        print(f"카메라 {camera_id} 등록 실패로 종료")
        return
    
    last_frame_time = datetime.now()
    frame_interval = 1.0 / camera_info["fps"]  # 프레임 간 시간 간격
    
    # 주기적 시간 동기화 설정
    last_sync_time = time.time()
    sync_interval = 60.0  # 60초마다 시간 동기화
    
    # 압축 설정
    print(f"카메라 {camera_id}의 이미지 압축 설정: 품질={quality}, 최대 폭={max_width}px")
    
    # 첫 프레임은 디버깅 정보 출력을 위해 True로 설정
    first_frame = True
    
    # WebSocket 핑/퐁 타이머
    last_ping_time = time.time()
    ping_interval = 30  # 30초마다 핑 전송
    
    try:
        while running:
            # 주기적으로 서버 시간 동기화
            current_time = time.time()
            if current_time - last_sync_time >= sync_interval:
                await sync_server_time()
                last_sync_time = current_time
                
            # 주기적으로 핑 전송 (연결 유지)
            if current_time - last_ping_time >= ping_interval:
                if camera_id in ws_connections:
                    try:
                        await ws_connections[camera_id].ping()
                        last_ping_time = current_time
                    except:
                        # 핑 실패 시 재연결 시도
                        if await connect_camera_websocket(camera_id, camera_info):
                            last_ping_time = current_time
            
            # 프레임 캡처
            ret, frame = camera.read()
            
            if not ret:
                print(f"카메라 {camera_id}로부터 프레임 읽기 실패")
                # 재연결 시도
                time.sleep(1)
                continue
            
            current_time = datetime.now()
            time_diff = (current_time - last_frame_time).total_seconds()
            
            # 지정된 FPS에 따라 이미지 업로드
            if time_diff >= frame_interval:
                # 이미지 인코딩 (압축 설정 적용)
                image_data = encode_image(frame, quality=quality, max_width=max_width, verbose=first_frame)
                
                # 첫 프레임 처리 후 verbose 플래그 끄기
                if first_frame:
                    first_frame = False
                
                if image_data:
                    # 서버 시간 기준으로 타임스탬프 생성
                    server_timestamp = get_server_time()
                    
                    # WebSocket을 통해 이미지 전송
                    if await send_frame_via_websocket(camera_id, image_data, server_timestamp):
                        last_frame_time = current_time
                    else:
                        # 전송 실패 시 재연결 시도
                        await connect_camera_websocket(camera_id, camera_info)
                    
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
        # WebSocket 연결 종료
        if camera_id in ws_connections:
            try:
                await ws_connections[camera_id].close()
                del ws_connections[camera_id]
            except:
                pass
        camera.release()

# 카메라 스레드 함수
def run_camera_thread(
    camera_id,
    camera_index,
    # resolution=(640, 480),
    fps=15,
    quality=85,
    max_width=640
):
    """개별 카메라 처리 스레드"""
    try:
        # 이벤트 루프 설정
        loop = get_event_loop()
        
        # 카메라 초기화
        # camera = init_camera(camera_index, resolution=resolution, fps=fps)
        camera = init_camera(camera_index, fps=fps)
        
        if not camera:
            print(f"카메라 {camera_id} 초기화 실패")
            return
        
        # 비동기 카메라 루프 실행 (압축 설정 전달)
        loop.run_until_complete(camera_loop(camera_id, camera, quality=quality, max_width=max_width))
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
    parser.add_argument('--quality', type=int, default=85,
                        help='이미지 압축 품질 (0-100, 기본값: 85)')
    parser.add_argument('--max-width', type=int, default=640,
                        help='이미지 최대 폭 (픽셀, 기본값: 640)')
    parser.add_argument('--fps', type=int, default=20,
                        help='카메라 프레임 레이트 (기본값: 20)')
    
    # 인자 파싱
    args = parser.parse_args()
    api_url = args.server
    
    # 압축 품질 범위 확인
    quality = max(1, min(100, args.quality))  # 1-100 범위로 제한
    if quality != args.quality:
        print(f"품질 값 조정: {args.quality} -> {quality} (유효 범위: 1-100)")
    
    # 최대 폭 확인
    max_width = max(320, args.max_width)  # 최소 320px
    
    print(f"프레임 레이트: {args.fps}")
    print(f"이미지 품질: {quality}")
    print(f"최대 이미지 폭: {max_width}px")
    
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
        # 카메라별 스레드 실행 함수 (클로저)
        def run_camera(camera_id, camera_index):
            return run_camera_thread(
                camera_id, 
                camera_index, 
                # resolution=(640, 480),  # 고정 해상도 사용
                fps=args.fps,
                quality=quality,
                max_width=max_width
            )
        
        thread = threading.Thread(
            target=run_camera,
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