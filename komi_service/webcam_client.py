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
import random
import numpy as np
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from PIL import ImageFont, ImageDraw, Image
# from pose_detection import YoloPoseModel

yolo_model = YOLO("yolo11x-pose.pt")
# 스레드별 전용 세션과 이벤트 루프
thread_local = threading.local()

# 전역 상태 변수
running = True
api_url = "http://localhost:8000"
server_time_offset = 0.0  # 서버와의 시간차 (초 단위)
ws_connections = {}  # 카메라 ID -> WebSocket 연결
connection_status = {}  # 카메라 ID -> 연결 상태 ("connected", "connecting", "disconnected")
last_ping_times = {}  # 카메라 ID -> 마지막 핑 전송 시간
# pose_model = None  # YOLO 포즈 모델 인스턴스
last_pose_detection_times = {}  # 카메라 ID -> 마지막 포즈 감지 시간
camera_status = {}  # 카메라 ID -> 상태 ("off", "on", "ready", "record", "detect")
video_recorders = {}  # 카메라 ID -> 비디오 레코더 객체
record_start_time = None

# 카메라 상태 정의
CAMERA_STATUS_OFF = "off"        # 초기상태, 이미지 캡쳐가 진행되지 않음
CAMERA_STATUS_ON = "on"          # 이미지 캡쳐가 진행되어 서버에 전달
CAMERA_STATUS_MASK = "mask"      # 이미지 캡쳐 및 후처리(반투명 mask 추가) 후 서버에 전달, 사용자 좌표 확인 후 ready로 상태 변경
CAMERA_STATUS_READY = "ready"    # 이미지 캡쳐 및 후처리(반투명 mask 추가) 후 서버에 전달
CAMERA_STATUS_RECORD = "record"  # 이미지 캡쳐 및 후처리(카운트 다운 및 녹화) 후 서버에 전달
CAMERA_STATUS_DETECT = "detect"  # 이미지 캡쳐 및 좌표 추출 후 서버에 전달

# WebSocket 연결 설정
MAX_RECONNECT_ATTEMPTS = 3  # 최대 재연결 시도 횟수
RECONNECT_DELAY = 2.0  # 초기 재연결 지연 시간(초)
PING_INTERVAL = 10  # 핑 전송 간격 (초)
FLIP_HORIZONTAL = False  # 좌우 반전 기본값
POSE_DETECTION_INTERVAL = 1.0  # 포즈 감지 간격 (초)

# 비디오 녹화 관련 상수
VIDEO_OUTPUT_DIR = "video_webcam"
VIDEO_FPS = 15
VIDEO_FORMAT = ".mp4"


def load_pose_database():
    """
    디렉토리 내의 모든 포즈 JSON을 로드하여 데이터베이스로 구성
    우선은 squat 포즈 데이터베이스만 로드
    Returns:
        np.ndarray: 포즈 데이터베이스, shape: (N, 68) - N은 파일 수, 각 행은 x1, y1, x2, y2, ... 형태로 펼쳐진 좌표
    """
    pose_database = {}
    pose_filenames = {}
    # 디렉토리 내 모든 JSON 파일 로드
    for direction in ['front', 'side']:
        pose_vectors = []
        filenames = []
        directory = f'data/squat/{direction}_json/'
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # 키포인트를 1차원 벡터로 변환 (x1, y1, x2, y2, ...)
            keypoints_flat = []
            for kp in data['keypoints']:
                keypoints_flat.append(kp["x"] if kp["x"] is not None else 0.0)
                keypoints_flat.append(kp["y"] if kp["y"] is not None else 0.0)
            
            pose_vectors.append(keypoints_flat)
            filenames.append(filename)
    
        # 전체 포즈 데이터베이스 생성
        pose_database[direction] = np.array(pose_vectors, dtype=np.float16)
        pose_filenames[direction] = filenames
        print(f"Loaded {len(pose_vectors)} pose vectors for {direction} direction")
    return pose_database, pose_filenames

# 전역 상수로 포즈 데이터베이스 로드
POSE_DATABASE, POSE_FILENAMES = load_pose_database()

def load_reference_pose(json_path):
    """기준 포즈 JSON을 로드하는 함수"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints = []
    for kp in data['keypoints']:
        if kp["x"] is not None and kp["y"] is not None:
            keypoints.append([kp["x"], kp["y"]])
        else:
            keypoints.append([None, None])
    return np.array(keypoints, dtype=np.float32)

REFERENCE_POSE = {
    "front": load_reference_pose('data/squat/front_json/frame_000.json'),
    "side": load_reference_pose('data/squat/side_json/frame_000.json'),
}


# 서버 시간 동기화 함수
async def sync_server_time():
    """서버 시간과 로컬 시간의 차이를 계산"""
    global server_time_offset
    try:
        session = await init_session()
        
        # 타임아웃 설정
        request_timeout = aiohttp.ClientTimeout(total=2)
        
        # 요청 전 로컬 시간 기록
        local_time_before = time.time()
        
        async with session.get(f"{api_url}/server_time", timeout=request_timeout) as response:
            if response.status != 200:
                return False
                
            # 응답 후 로컬 시간 기록
            local_time_after = time.time()
            data = await response.json()
            
            server_timestamp = data.get("timestamp")
            if not server_timestamp:
                return False
            
            # 네트워크 지연 시간 추정 (왕복 시간의 절반)
            network_delay = (local_time_after - local_time_before) / 2
            
            # 보정된 서버 시간과 로컬 시간의 차이 계산
            local_time_avg = local_time_before + network_delay
            server_time_offset = server_timestamp - local_time_avg
            return True
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
        # 향상된 타임아웃 설정으로 세션 생성
        timeout = aiohttp.ClientTimeout(total=30, connect=5, sock_connect=5, sock_read=5)
        thread_local.session = aiohttp.ClientSession(timeout=timeout)
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
def init_camera(camera_index, fps=15):
    """카메라 초기화 및 설정"""
    try:
        # 카메라 객체 생성
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # FPS 설정
        if fps:
            camera.set(cv2.CAP_PROP_FPS, fps)
        
        # 카메라 연결 확인
        if not camera.isOpened():
            print(f"카메라 {camera_index} 연결 실패")
            return None
            
        # 설정 확인
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        
        print(f"카메라 {camera_index} 초기화 완료:")
        print(f"  - 요청 FPS: {fps}")
        print(f"  - 실제 FPS: {actual_fps}")
        
        return camera
    except Exception as e:
        print(f"카메라 초기화 오류: {str(e)}")
        return None

# 카메라 상태 변경 함수
async def set_camera_status(camera_id, new_status):
    """
    서버에 카메라 상태 변경을 요청하는 함수
    
    Args:
        camera_id (str): 카메라 ID
        new_status (str): 새로운 상태 (off, on, mask, ready, record, detect)
    
    Returns:
        bool: 성공 여부
    """
    global camera_status
    
    try:
        # 세션 초기화
        session = await init_session()
        
        # 서버에 상태 변경 요청
        async with session.post(
            f"{api_url}/cameras/{camera_id}/status",
            json={"status": new_status}
        ) as response:
            if response.status == 200:
                # 서버 응답 확인
                result = await response.json()
                
                # 로컬 상태 업데이트
                camera_status[camera_id] = new_status
                print(f"카메라 {camera_id} 상태 변경 요청 성공: {new_status}")
                return True
            else:
                error_text = await response.text()
                print(f"카메라 {camera_id} 상태 변경 요청 실패 (상태 코드: {response.status}): {error_text}")
                return False
                
    except Exception as e:
        print(f"카메라 {camera_id} 상태 변경 요청 중 오류: {str(e)}")
        return False

# 이미지 인코딩 함수
def encode_image(frame, quality=85, max_width=640, flip=False, verbose=False):
    """이미지를 Base64 인코딩"""
    try:
        # 좌우 반전 처리
        if flip:
            frame = cv2.flip(frame, 1)  # 1은 좌우 반전을 의미
            
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
        
        # JPEG으로 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        
        # Base64로 인코딩
        base64_image = base64.b64encode(encimg).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"이미지 인코딩 오류: {str(e)}")
        return None

# 연결 상태 업데이트 함수
def update_connection_status(camera_id, status):
    """카메라 연결 상태 업데이트"""
    old_status = connection_status.get(camera_id)
    connection_status[camera_id] = status
    
    if status == "connected" and old_status != "connected":
        print(f"카메라 {camera_id} 연결 성공")
    elif status == "connecting" and old_status != "connecting":
        print(f"카메라 {camera_id} 연결 시도 중...")
    elif status == "disconnected" and old_status != "disconnected":
        print(f"카메라 {camera_id} 연결 끊김")

# WebSocket 연결 종료 함수
async def close_camera_connection(camera_id):
    """WebSocket 연결을 정상적으로 종료하고 서버에 알림"""
    if camera_id in ws_connections:
        try:
            # 종료 메시지 전송
            ws = ws_connections[camera_id]
            disconnect_msg = {
                "type": "disconnect",
                "camera_id": camera_id,
                "reason": "client_shutdown"
            }
            await ws.send_json(disconnect_msg)
            
            # 약간의 지연을 두어 메시지가 전송될 시간 확보
            await asyncio.sleep(0.5)
            
            # 연결 종료
            await ws.close()
            print(f"카메라 {camera_id} 연결 정상 종료")
        except Exception as e:
            print(f"카메라 {camera_id} 연결 종료 중 오류: {str(e)}")
        finally:
            # 상태 업데이트 및 연결 정보 삭제
            if camera_id in ws_connections:
                del ws_connections[camera_id]
            update_connection_status(camera_id, "disconnected")

# WebSocket 연결 및 카메라 등록
async def connect_camera_websocket(camera_id, camera_info, retry_count=0):
    """WebSocket을 통해 카메라 연결 및 등록"""
    global camera_status
    
    # 연결 상태 업데이트
    update_connection_status(camera_id, "connecting")
    
    # 카메라 상태가 없으면 기본 ON 상태로 설정
    if camera_id not in camera_status:
        camera_status[camera_id] = CAMERA_STATUS_ON
    
    try:
        session = await init_session()
        
        # WebSocket URL 생성
        if api_url.startswith('http://'):
            ws_url = f"ws://{api_url[7:]}/ws/camera"
        elif api_url.startswith('https://'):
            ws_url = f"wss://{api_url[8:]}/ws/camera"
        else:
            ws_url = f"ws://{api_url}/ws/camera"
        
        # WebSocket 연결 설정 - 자동 heartbeat 비활성화
        ws = await session.ws_connect(
            ws_url, 
            timeout=aiohttp.ClientWSTimeout(ws_close=60.0),
            heartbeat=None,
            max_msg_size=0
        )
        
        # 카메라 등록 메시지 전송 - 상태 정보 추가
        register_msg = {
            "type": "register",
            "camera_id": camera_id,
            "info": camera_info,
            "status": camera_status[camera_id]  # 현재 상태 정보 추가
        }
        
        await ws.send_json(register_msg)
        
        # 응답 대기 (타임아웃 설정)
        response = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
        
        if response.get("type") == "connection_successful":
            # 연결 성공시 상태 업데이트 및 정보 저장
            ws_connections[camera_id] = ws
            update_connection_status(camera_id, "connected")
            last_ping_times[camera_id] = time.time()
            
            # 서버에 현재 상태 명시적으로 알림
            await ws.send_json({
                "type": "status_changed",
                "camera_id": camera_id,
                "status": camera_status[camera_id]
            })
            
            # 서버 메시지 처리 태스크 시작
            asyncio.create_task(handle_server_messages(camera_id, ws))
            return True
        else:
            update_connection_status(camera_id, "disconnected")
            await ws.close()
            return False
            
    except Exception as e:
        update_connection_status(camera_id, "disconnected")
        print(f"WebSocket 연결 오류 ({retry_count}): {str(e)}")
        
        # 재시도 횟수가 MAX_RECONNECT_ATTEMPTS 미만인 경우 재연결 시도
        if retry_count < MAX_RECONNECT_ATTEMPTS:
            # 지수 백오프로 재시도 간격 증가
            delay = RECONNECT_DELAY * (2 ** retry_count) + random.uniform(0, 1.0)
            print(f"카메라 {camera_id} {delay:.1f}초 후 재연결 시도...")
            await asyncio.sleep(delay)
            return await connect_camera_websocket(camera_id, camera_info, retry_count + 1)
        return False

# 서버 메시지 처리 함수
async def handle_server_messages(camera_id, ws):
    """서버의 WebSocket 메시지를 처리하는 함수"""
    global camera_status, video_recorders
    
    # 초기 상태는 켜짐
    if camera_id not in camera_status:
        camera_status[camera_id] = CAMERA_STATUS_ON
    
    try:
        # 연결이 유지되는 동안 반복
        while camera_id in ws_connections and connection_status.get(camera_id) == "connected":
            try:
                # 짧은 타임아웃으로 메시지 수신
                msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
                
                # 텍스트 메시지 처리
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # 핑/퐁 메시지 처리
                    if msg.data == "ping":
                        await ws.send_str("pong")  # 퐁으로 응답
                        last_ping_times[camera_id] = time.time()
                    elif msg.data == "pong":
                        last_ping_times[camera_id] = time.time()
                    else:
                        # JSON 메시지 처리
                        try:
                            data = json.loads(msg.data)
                            # 상태 제어 메시지 처리
                            if data.get("type") == "status_control":
                                old_status = camera_status.get(camera_id, CAMERA_STATUS_ON)
                                new_status = data.get("status", CAMERA_STATUS_ON)
                                
                                # 상태 변경 확인
                                if old_status != new_status:
                                    # 녹화 중 상태에서 다른 상태로 변경 시 녹화 종료
                                    if old_status == CAMERA_STATUS_RECORD and new_status != CAMERA_STATUS_RECORD:
                                        video_path, duration = stop_video_recording(camera_id)
                                        
                                        # 녹화 파일이 존재하면 서버에 업로드
                                        if video_path and duration > 0:
                                            # 비동기 업로드 태스크 시작
                                            asyncio.create_task(upload_video_to_server(camera_id, video_path, duration))
                                    
                                    # 상태 업데이트
                                    camera_status[camera_id] = new_status
                                    print(f"카메라 {camera_id} 상태 변경: {old_status} -> {new_status}")
                                    
                                    # 서버에 상태 변경 확인 메시지 전송
                                    await ws.send_json({
                                        "type": "status_changed",
                                        "camera_id": camera_id,
                                        "status": new_status
                                    })
                        except json.JSONDecodeError:
                            pass
                # 연결 종료 메시지 처리
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
            except asyncio.TimeoutError:
                pass  # 타임아웃은 정상, 계속 진행
            except Exception as e:
                print(f"서버 메시지 처리 오류: {str(e)}")
                break
    finally:
        # 연결 종료 시 녹화 중이면 녹화 중지
        if camera_id in camera_status and camera_status[camera_id] == CAMERA_STATUS_RECORD:
            video_path, duration = stop_video_recording(camera_id)
            # 녹화 파일이 있으면 서버에 업로드 시도
            if video_path and duration > 0:
                try:
                    loop = get_event_loop()
                    loop.run_until_complete(upload_video_to_server(camera_id, video_path, duration))
                except Exception as e:
                    print(f"연결 종료 시 비디오 업로드 오류: {str(e)}")
            
        if camera_id in connection_status and connection_status[camera_id] == "connected":
            update_connection_status(camera_id, "disconnected")
            if camera_id in ws_connections:
                del ws_connections[camera_id]

# WebSocket을 통한 이미지 전송
async def send_frame_via_websocket(camera_id, image_data, timestamp, pose_data=None):
    """WebSocket을 통해 이미지 프레임 전송"""
    if camera_id not in ws_connections:
        return False
    
    try:
        ws = ws_connections[camera_id]
        
        # 프레임 메시지 생성
        frame_msg = {
            "type": "frame",
            "camera_id": camera_id,
            "image_data": image_data,
            "timestamp": timestamp.isoformat()
        }
        
        # 포즈 데이터가 있으면 추가
        if pose_data:
            frame_msg["pose_data"] = pose_data
            frame_msg["type"] = "frame_with_pose"
        
        # 메시지 전송
        await ws.send_json(frame_msg)
        return True
            
    except Exception as e:
        print(f"이미지 전송 오류: {str(e)}")
        
        # 연결 오류 시 상태 업데이트
        if camera_id in ws_connections:
            del ws_connections[camera_id]
        update_connection_status(camera_id, "disconnected")
        return False

# 메인 카메라 처리 루프
async def camera_loop(camera_id, camera, quality=85, max_width=640, flip=False):
    """단일 카메라 처리 비동기 루프"""
    global camera_status, last_pose_detection_times
    
    # # 포즈 감지 모델이 없으면 초기화
    # if pose_model is None:
    #     pose_model = YoloPoseModel()
    
    # 카메라 정보 구성
    camera_info = {
        "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(camera.get(cv2.CAP_PROP_FPS))
    }
    
    # 서버 시간 동기화
    await sync_server_time()
    
    # 프레임 속도 계산
    fps = camera_info["fps"]
    frame_interval = 1.0 / fps
    
    # 주기적 동기화 변수
    last_sync_time = time.time()
    last_ping_time = time.time()
    last_frame_time = datetime.now()
    
    # 포즈 감지 시간 초기화
    last_pose_detection_times[camera_id] = time.time() - POSE_DETECTION_INTERVAL  # 시작 시 바로 감지하도록 설정
    
    # 카메라 상태 초기화
    camera_status[camera_id] = CAMERA_STATUS_ON
    
    # 최초 웹소켓 연결
    if not await connect_camera_websocket(camera_id, camera_info):
        print(f"카메라 {camera_id} 초기 연결 실패")
    
    try:
        while running:
            # 연결이 끊어진 경우 재연결 시도
            if camera_id not in ws_connections:
                if not await connect_camera_websocket(camera_id, camera_info):
                    # 연결 실패 시 잠시 대기
                    await asyncio.sleep(2.0)
                    continue
            
            # 주기적으로 서버 시간 동기화 (5분마다)
            current_time = time.time()
            if current_time - last_sync_time >= 300:
                await sync_server_time()
                last_sync_time = current_time
            
            # 주기적으로 핑 전송 (연결 유지)
            if current_time - last_ping_time >= PING_INTERVAL:
                if camera_id in ws_connections:
                    try:
                        await ws_connections[camera_id].ping()
                        last_ping_time = current_time
                    except Exception:
                        # 연결 끊어짐
                        if camera_id in ws_connections:
                            del ws_connections[camera_id]
                        update_connection_status(camera_id, "disconnected")
                        continue
            
            # 현재 카메라 상태 확인
            current_status = camera_status.get(camera_id, CAMERA_STATUS_ON)
            
            # OFF 상태면 프레임 처리 건너뛰기
            if current_status == CAMERA_STATUS_OFF:
                await asyncio.sleep(0.5)  # 주기적으로 확인
                continue
            
            # 프레임 캡처
            ret, frame = camera.read()
            
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            # 현재 시간과 마지막 프레임 전송 시간의 차이 계산
            current_time = datetime.now()
            time_diff = (current_time - last_frame_time).total_seconds()
            
            # 지정된 FPS에 따라 이미지 업로드
            if time_diff >= frame_interval:
                # 서버 시간 기준으로 타임스탬프 생성
                server_timestamp = get_server_time()
                
                # 상태에 따른 프레임 처리
                if await process_frame_by_status(
                    camera_id, frame, server_timestamp, current_status, 
                    quality=quality, max_width=max_width, flip=flip
                ):
                    last_frame_time = current_time
            
            # 적절한 딜레이
            remaining_time = frame_interval - (datetime.now() - current_time).total_seconds()
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
            else:
                await asyncio.sleep(0.001)  # 최소 딜레이
                
    except asyncio.CancelledError:
        # 작업 취소됨 (정상)
        pass
    except Exception as e:
        print(f"카메라 {camera_id} 루프 오류: {str(e)}")
    finally:
        # 녹화 중이면 녹화 중지
        if camera_id in camera_status and camera_status[camera_id] == CAMERA_STATUS_RECORD:
            video_path, duration = stop_video_recording(camera_id)
            # 녹화 파일이 있으면 서버에 업로드 시도
            if video_path and duration > 0:
                try:
                    loop = get_event_loop()
                    loop.run_until_complete(upload_video_to_server(camera_id, video_path, duration))
                except Exception as e:
                    print(f"연결 종료 시 비디오 업로드 오류: {str(e)}")
            
        # 연결 종료 및 리소스 정리
        await close_camera_connection(camera_id)
        camera.release()
        print(f"카메라 {camera_id} 리소스 해제 완료")

# 카메라 스레드 함수
def run_camera_thread(camera_id, camera_index, fps=15, quality=85, max_width=640, flip=False):
    """개별 카메라 처리 스레드"""
    try:
        # 이벤트 루프 설정
        loop = get_event_loop()
        
        # 카메라 초기화
        camera = init_camera(camera_index, fps=fps)
        
        if not camera:
            print(f"카메라 {camera_id} 초기화 실패")
            return
        
        # 비동기 카메라 루프 실행
        loop.run_until_complete(camera_loop(camera_id, camera, quality=quality, max_width=max_width, flip=flip))
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

# 비디오 녹화 시작 함수
def start_video_recording(camera_id, frame_width, frame_height, fps=VIDEO_FPS):
    """비디오 녹화 시작"""
    global video_recorders
    
    # 출력 디렉토리 생성
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    
    # 현재 시간으로 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(VIDEO_OUTPUT_DIR, f"{camera_id}_{timestamp}{VIDEO_FORMAT}")
    
    # OpenCV VideoWriter 생성
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
    # fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 코덱 (스트림릿 호환)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 코덱 (openh264-1.8.0-win64.dll 파일 필요)
    
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    if not video_writer.isOpened():
        print(f"카메라 {camera_id} 비디오 녹화 초기화 실패")
        return None
    
    video_recorders[camera_id] = {
        "writer": video_writer,
        "path": video_path,
        "start_time": datetime.now()
    }
    
    print(f"카메라 {camera_id} 비디오 녹화 시작: {video_path}")
    return video_writer

# 비디오 녹화 중지 함수
def stop_video_recording(camera_id):
    """비디오 녹화 중지"""
    global video_recorders
    
    if camera_id in video_recorders and video_recorders[camera_id]["writer"]:
        # 녹화 정보 저장
        recorder_info = video_recorders[camera_id]
        video_writer = recorder_info["writer"]
        
        # 리소스 해제
        video_writer.release()
        
        # 녹화 시간 계산
        duration = (datetime.now() - recorder_info["start_time"]).total_seconds()
        
        print(f"카메라 {camera_id} 비디오 녹화 종료: {recorder_info['path']} ({duration:.1f}초)")
        
        # 레코더 정보에서 writer 제거 (나머지 정보는 유지)
        video_recorders[camera_id]["writer"] = None
        
        return recorder_info["path"], duration
    
    return None, 0

# 영상 파일 업로드 함수
async def upload_video_to_server(camera_id, video_path, duration):
    """녹화된 비디오를 서버에 업로드"""
    if not os.path.exists(video_path):
        print(f"업로드할 비디오 파일을 찾을 수 없습니다: {video_path}")
        return False
        
    try:
        session = await init_session()
        
        # 업로드할 파일 준비
        with open(video_path, 'rb') as f:
            file_data = f.read()
        
        # 파일명을 카메라 ID로 설정 (확장자 유지)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{camera_id}_{timestamp}.mp4"
        
        # multipart/form-data 요청 생성
        data = aiohttp.FormData()
        data.add_field('video', 
                      file_data,
                      filename=file_name,
                      content_type='video/mp4')
        data.add_field('exercise_id', 'webcam_recording')
        data.add_field('user_id', camera_id)
        data.add_field('camera_id', camera_id)  # 카메라 ID도 추가로 전송
        
        # 업로드 요청 전송
        upload_timeout = aiohttp.ClientTimeout(total=300)  # 5분 타임아웃 (큰 파일 고려)
        async with session.post(
            f"{api_url}/videos/upload",
            data=data,
            timeout=upload_timeout
        ) as response:
            if response.status in (200, 201):
                result = await response.json()
                print(f"비디오 업로드 성공: {result.get('video_id')}")
                
                # 영상 정보를 서버에 추가로 전송
                await send_recording_info(camera_id, result.get('video_id'), video_path, duration)
                return True
            else:
                error_text = await response.text()
                print(f"비디오 업로드 실패 (상태 코드: {response.status}): {error_text}")
                return False
                
    except Exception as e:
        print(f"비디오 업로드 오류: {str(e)}")
        return False

# 서버에 녹화 정보 전송
async def send_recording_info(camera_id, video_id, video_path, duration):
    """서버에 녹화 완료 정보 전송"""
    if camera_id not in ws_connections:
        print(f"카메라 {camera_id}의 웹소켓 연결이 없어 녹화 정보를 전송할 수 없습니다")
        return False
    
    try:
        ws = ws_connections[camera_id]
        
        # 녹화 완료 메시지 생성
        recording_info = {
            "type": "recording_completed",
            "camera_id": camera_id,
            "video_id": video_id,
            "video_path": video_path,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        
        # 메시지 전송
        await ws.send_json(recording_info)
        print(f"카메라 {camera_id} 녹화 정보 전송 완료")
        return True
        
    except Exception as e:
        print(f"녹화 정보 전송 오류: {str(e)}")
        return False
    
def put_korean_text(image, text, position, font_size=30, color=(0, 0, 0), thickness=1):
    """한글 텍스트를 이미지에 추가하는 함수"""
    # OpenCV 이미지를 PIL 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # PIL 이미지에 텍스트 그리기
    draw = ImageDraw.Draw(pil_image)
    
    # 폰트 설정 (윈도우의 경우 시스템 폰트 경로)
    # 맑은 고딕 폰트가 없는 경우, 다른 한글 폰트로 대체 가능
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)  # Windows 기본 한글 폰트
    except IOError:
        try:
            font = ImageFont.truetype("NanumGothic.ttf", font_size)  # 나눔고딕 폰트
        except IOError:
            try:
                # Ubuntu 등 리눅스 시스템 폰트
                font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
            except IOError:
                # 폰트를 찾을 수 없는 경우 기본 폰트 사용
                font = ImageFont.load_default()
    
    # 텍스트 줄바꿈 처리
    lines = text.split('\n')
    line_height = font_size + 5  # 줄 간격
    
    # 각 줄 그리기
    for i, line in enumerate(lines):
        pos = (position[0], position[1] + i * line_height)
        # 한글 텍스트 그리기
        draw.text(pos, line, font=font, fill=color)
    
    # PIL 이미지를 OpenCV 이미지로 다시 변환
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result_image

# 마스크 오버레이 함수 (반투명 적용)
def overlay_mask(frame, mask, alpha_value=100):
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    # if mask_resized.shape[2] == 3:  # RGB
    #     mask_rgb = mask_resized
    #     mask_alpha = np.ones((mask_resized.shape[0], mask_resized.shape[1]), dtype=np.uint8) * 255
    # else:  # RGBA
    #     mask_rgb = mask_resized[:, :, :3]
    #     mask_alpha = mask_resized[:, :, 3]
    mask_rgb = mask_resized[:, :, :3]
    mask_alpha = mask_resized[:, :, 3]
    object_mask = (mask_alpha > 0).astype(np.uint8)
    custom_alpha = np.full_like(mask_alpha, alpha_value, dtype=np.uint8)
    custom_alpha[object_mask == 0] = 0
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    for c in range(3):
        frame_rgba[:, :, c] = (
            frame_rgba[:, :, c] * (1 - custom_alpha / 255.0) +
            mask_rgb[:, :, c] * (custom_alpha / 255.0)
        ).astype(np.uint8)
    frame_rgba[:, :, 3] = np.maximum(frame_rgba[:, :, 3], custom_alpha)
    return cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

# 카운트다운 오버레이 함수
def overlay_countdown(frame, remaining):
    """카운트다운 숫자를 프레임에 오버레이하고 수정된 프레임을 반환합니다"""
    # 좌우 반전 적용
    frame = cv2.flip(frame, 1)
    
    # 프레임 크기 가져오기
    height, width = frame.shape[:2]
    # 텍스트 크기 계산
    text = str(remaining)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)[0]
    # 중앙 상단에 위치하도록 좌표 계산
    text_x = (width - text_size[0]) // 2
    text_y = text_size[1] + 50  # 상단에서 약간 여백 추가
    # 텍스트 그리기
    result_frame = cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
    return result_frame


cnt_tmp_test = 0
def is_pose_similar_by_accuracy(current_pose, camera_id, threshold_px=20, ratio=0.7):
    try:
        reference_pose = REFERENCE_POSE[camera_id]
        
        # 입력 검증
        if current_pose is None or len(current_pose) == 0:
            return False
            
        if reference_pose is None or len(reference_pose) == 0:
            return False
            
        # 배열 크기 확인
        if len(current_pose) != len(reference_pose):
            print(f"포즈 배열 크기 불일치: current={len(current_pose)}, reference={len(reference_pose)}")
            return False
            
        match_count = 0
        total_count = 0
        
        for i in range(len(reference_pose)):
            try:
                ref = reference_pose[i]
                cur = current_pose[i]
                
                if ref[0] is None or cur[0] is None:
                    continue
                    
                if np.isnan(cur[0]) or np.isnan(cur[1]):
                    continue
                    
                dist = np.linalg.norm(np.array(ref) - np.array(cur))
                total_count += 1
                if dist <= threshold_px:
                    match_count += 1
                    
            except Exception as e:
                print(f"키포인트 {i} 비교 중 오류: {str(e)}")
                continue
                
        if total_count == 0:
            return False
            
        global cnt_tmp_test
        cnt_tmp_test += 1
        if cnt_tmp_test % 30 == 0:
            print(f"정확도: {match_count / total_count:.2f}")
            print(datetime.now())
            
        return (match_count / total_count) >= ratio
        
    except Exception as e:
        print(f"포즈 유사도 계산 중 오류: {str(e)}")
        return False


def check_pose_alignment(frame, yolo_model, camera_id):
    try:
        results = yolo_model.predict(source=frame, stream=False, verbose=False)
        if not results or len(results) == 0:
            return False
            
        result = results[0]
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return False
            
        keypoints_data = result.keypoints.xy
        if len(keypoints_data) == 0:
            return False
            
        try:
            keypoints = keypoints_data[0].cpu().numpy()
        except IndexError:
            return False
            
        if len(keypoints) == 0:
            return False
            
        keypoints_array = np.array(keypoints, dtype=np.float32)
        return is_pose_similar_by_accuracy(keypoints_array, camera_id)
        
    except Exception as e:
        print(f"포즈 정렬 확인 중 오류: {str(e)}")
        return False

def analyze_pose(keypoints: np.ndarray) -> list[str]:
    """
    YOLO 기반 키포인트 배열을 받아 스쿼트 동작을 분석하고 피드백 제공
    """
    # 입력 검증 강화
    if keypoints is None or not isinstance(keypoints, np.ndarray) or keypoints.size == 0:
        return ["포즈가 감지되지 않았습니다."]
    
    # 주요 관절 인덱스
    p_idx = {
        "l_shoulder": 5, "r_shoulder": 6,
        "l_hip": 11, "r_hip": 12,
        "l_knee": 13, "r_knee": 14,
        "l_ankle": 15, "r_ankle": 16,
    }

    def get_point(name):
        try:
            i = p_idx.get(name)
            if i is None or i >= len(keypoints):
                return None
            x, y = keypoints[i]
            if np.isnan(x) or np.isnan(y):
                return None
            return np.array([float(x), float(y)])
        except (IndexError, ValueError, TypeError) as e:
            print(f"키포인트 추출 중 오류 발생: {str(e)}")
            return None

    def compute_angle(p1, p2, p3):
        try:
            if any(p is None for p in [p1, p2, p3]):
                return None
            
            v1, v2 = p1 - p2, p3 - p2
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            
            if norm1 < 1e-10 or norm2 < 1e-10:  # 거의 0에 가까운 경우 체크
                return None

            cosine = np.dot(v1, v2) / (norm1 * norm2)
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        except (ValueError, TypeError, ZeroDivisionError) as e:
            print(f"각도 계산 중 오류 발생: {str(e)}")
            return None

    def safe_vector_operation(p1, p2, operation="distance"):
        try:
            if p1 is None or p2 is None:
                return None
            if not isinstance(p1, np.ndarray) or not isinstance(p2, np.ndarray):
                return None
            if p1.size == 0 or p2.size == 0:
                return None
            if p1.shape != (2,) or p2.shape != (2,):  # 2D 포인트 확인
                return None
                
            if operation == "distance":
                diff = p2 - p1
                if diff.size == 0:
                    return None
                return float(np.linalg.norm(diff))
            elif operation == "midpoint":
                return (p1 + p2) / 2
            return None
        except (ValueError, TypeError) as e:
            print(f"벡터 연산 중 오류 발생: {str(e)}")
            return None

    # 포인트 가져오기
    points = {name: get_point(name) for name in p_idx.keys()}
    
    # 모든 포인트가 None인 경우 early return
    if all(p is None for p in points.values()):
        return ["포즈가 제대로 감지되지 않았습니다. 카메라 위치를 조정해주세요."]

    # 관절 각도 계산
    angles = {}
    
    # 안전하게 각도 계산
    try:
        angles["left_hip_angle"] = compute_angle(points["l_shoulder"], points["l_hip"], points["l_knee"])
        angles["right_hip_angle"] = compute_angle(points["r_shoulder"], points["r_hip"], points["r_knee"])
        angles["left_knee_angle"] = compute_angle(points["l_hip"], points["l_knee"], points["l_ankle"])
        angles["right_knee_angle"] = compute_angle(points["r_hip"], points["r_knee"], points["r_ankle"])
        
        # 어깨와 엉덩이 너비 계산
        if all(points[p] is not None for p in ["l_shoulder", "r_shoulder"]):
            angles["shoulder_width"] = safe_vector_operation(points["l_shoulder"], points["r_shoulder"], "distance")
        
        if all(points[p] is not None for p in ["l_hip", "r_hip"]):
            angles["hip_width"] = safe_vector_operation(points["l_hip"], points["r_hip"], "distance")
        
        # 무릎 간격 계산
        if all(points[p] is not None for p in ["l_knee", "r_knee"]):
            angles["knee_to_knee_distance"] = safe_vector_operation(points["l_knee"], points["r_knee"], "distance")
        
        # 척추 각도 계산
        if all(points[p] is not None for p in ["l_shoulder", "r_shoulder", "l_hip", "r_hip"]):
            mid_sh = safe_vector_operation(points["l_shoulder"], points["r_shoulder"], "midpoint")
            mid_hip = safe_vector_operation(points["l_hip"], points["r_hip"], "midpoint")
            if mid_sh is not None and mid_hip is not None:
                vertical = mid_hip + np.array([0, -100])
                angles["back_angle"] = compute_angle(mid_sh, mid_hip, vertical)
    except Exception as e:
        print(f"각도 계산 중 오류 발생: {str(e)}")
        return ["포즈 분석 중 오류가 발생했습니다."]

    # 피드백 생성
    feedback = []
    
    # 각도가 계산되지 않은 경우 기본 메시지 추가
    if not angles:
        return ["자세가 정확하게 감지되지 않았습니다. 카메라 앞에서 전신이 보이도록 서주세요."]

    # 기존 피드백 로직
    if angles.get("left_knee_angle") is not None and 100 > angles["left_knee_angle"] > 90:
        feedback.append("왼쪽 무릎을 더 굽혀주세요.")
    if angles.get("right_knee_angle") is not None and 100 > angles["right_knee_angle"] > 90:
        feedback.append("오른쪽 무릎을 더 굽혀주세요.")
    
    if angles.get("left_hip_angle") is not None and 130 > angles["left_hip_angle"] > 120:
        feedback.append("왼쪽 엉덩이를 더 낮춰보세요.")
    if angles.get("right_hip_angle") is not None and 130 > angles["right_hip_angle"] > 120:
        feedback.append("오른쪽 엉덩이를 더 낮춰보세요.")
    
    if angles.get("back_angle") is not None and angles["back_angle"] < 70:
        feedback.append("허리를 너무 숙이지 마세요.")

    knee_to_knee = angles.get("knee_to_knee_distance")
    width = angles.get("shoulder_width")
    if knee_to_knee is not None and width is not None:
        if width * 0.5 < knee_to_knee < width * 0.8:
            feedback.append("무릎이 너무 붙었습니다. 벌려주세요.")
        elif width * 1.2 < knee_to_knee < width * 1.5:
            feedback.append("무릎이 너무 벌어졌습니다. 모아주세요.")

    return feedback


def check_similar_pose(frame, camera_id=None):
    """
    포즈 감지 및 가장 유사한 포즈 파일명 반환
    """
    global yolo_model, POSE_DATABASE, POSE_FILENAMES
    
    # 포즈 방향 결정
    direction = camera_id if camera_id in ['front', 'side'] else 'front'
    
    try:
        # 프레임에서 포즈 감지
        results = yolo_model.predict(source=frame, stream=False, verbose=False)
        
        # 감지된 사람이 있는지 확인
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                # 첫 번째 사람의 키포인트 추출
                keypoints = result.keypoints.xy[0].cpu().numpy()
                
                # 유효하지 않은 키포인트 확인
                if keypoints.size == 0:
                    return POSE_FILENAMES[direction][0]  # 기본값 반환
                
                # 키포인트를 1차원 벡터로 변환
                keypoints_flat = []
                for kp in keypoints:
                    x = float(kp[0]) if not np.isnan(kp[0]) else 0.0
                    y = float(kp[1]) if not np.isnan(kp[1]) else 0.0
                    keypoints_flat.extend([x, y])
                
                # 모든 값이 0인지 확인
                if not any(keypoints_flat):
                    return POSE_FILENAMES[direction][0]  # 기본값 반환
                
                # 사용자 포즈 벡터 생성
                user_pose = np.array(keypoints_flat).reshape(1, -1)
                
                # 데이터베이스 크기 확인
                if POSE_DATABASE[direction].size == 0:
                    return POSE_FILENAMES[direction][0]  # 기본값 반환
                
                # 차원 일치 확인
                if user_pose.shape[1] != POSE_DATABASE[direction].shape[1]:
                    print(f"차원 불일치: 사용자 포즈 {user_pose.shape[1]}, DB {POSE_DATABASE[direction].shape[1]}")
                    return POSE_FILENAMES[direction][0]  # 기본값 반환
                
                try:
                    # 코사인 유사도 계산
                    similarities = cosine_similarity(user_pose, POSE_DATABASE[direction])
                    most_similar_index = np.argmax(similarities)
                    return POSE_FILENAMES[direction][most_similar_index]
                except Exception as e:
                    print(f"유사도 계산 오류: {str(e)}")
                    return POSE_FILENAMES[direction][0]  # 기본값 반환
                
        return POSE_FILENAMES[direction][0]  # 포즈가 감지되지 않은 경우 기본값 반환
        
    except Exception as e:
        print(f"포즈 체크 중 오류 발생: {str(e)}")
        return POSE_FILENAMES[direction][0]  # 오류 발생 시 기본값 반환

# 프레임 후처리 함수
def post_process_mask(frame, camera_id=None):
    """프레임 후처리 (상태에 따라 다른 처리)"""
    mask_image_path = f'data/squat/{camera_id}_frame_000_mask.png'
    mask = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    return overlay_mask(frame, mask)


def post_process_record(frame, camera_id=None, remaining=0):
    """프레임 후처리 (상태에 따라 다른 처리)"""
    if remaining > 0:
        return overlay_countdown(frame, remaining)
    elif remaining > -2.9:
        # 녹화 상태 처리
        if camera_id not in video_recorders or not video_recorders.get(camera_id, {}).get("writer"):
            # 녹화 중이 아니면 녹화 시작
            height, width = frame.shape[:2]
            start_video_recording(camera_id, width, height)
        
        # 비디오에 프레임 저장
        if camera_id in video_recorders and video_recorders[camera_id].get("writer"):
            video_recorders[camera_id]["writer"].write(frame)
    return frame

def post_process_detect(frame, camera_id=None, threshold_px=20):
    """
    포즈 감지 후처리 - 가장 유사한 포즈와 비교하여 문제 지점 시각화
    중요한 키포인트만 출력
    
    Args:
        frame: 입력 영상 프레임
        camera_id: 카메라 ID (front/side 구분용)
        
    Returns:
        시각화된 결과 프레임
    """
    global yolo_model, POSE_DATABASE, POSE_FILENAMES
    
    frame = cv2.flip(frame, 1)
    direction = camera_id if camera_id in ['front', 'side'] else 'front'
    result_frame = frame.copy()
    
    try:
        # 프레임에서 포즈 감지
        results = yolo_model.predict(source=frame, stream=False, verbose=False)
        
        # 결과가 비어있는지 확인
        if not results or len(results) == 0:
            return result_frame
            
        result = results[0]
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return result_frame
            
        # keypoints가 비어있는지 확인
        keypoints_data = result.keypoints.xy
        if len(keypoints_data) == 0:
            return result_frame
            
        # 첫 번째 사람의 키포인트 추출 시도
        try:
            keypoints = keypoints_data[0].cpu().numpy()
        except IndexError:
            return result_frame
            
        if len(keypoints) == 0:
            return result_frame
            
        # 가장 유사한 포즈 파일명 찾기
        best_match_filename = check_similar_pose(frame, camera_id=direction)
        
        # 기준 포즈 JSON 로드
        reference_file_path = f'data/squat/{direction}_json/{best_match_filename}'
        with open(reference_file_path, 'r') as f:
            reference_data = json.load(f)
        
        # 기준 포즈의 키포인트 추출
        reference_keypoints = []
        for kp in reference_data['keypoints']:
            if kp["x"] is not None and kp["y"] is not None:
                reference_keypoints.append([kp["x"], kp["y"]])
            else:
                reference_keypoints.append([None, None])
        
        # 얼굴 관련 키포인트 인덱스 (YOLO 포맷)
        # 0: 코, 1-4: 얼굴(왼쪽 눈, 오른쪽 눈, 왼쪽 귀, 오른쪽 귀)
        # face_keypoints = [1, 2, 3, 4]  # 코(0)는 제외, 표시해야 함
        # 중요한 키포인트만 추출
        core_keypoints = [0, 5, 6, 11, 12, 13, 14, 15, 16]
        # 거리 계산 및 시각화
        for i in core_keypoints:
            # # 얼굴 관련 키포인트는 코를 제외하고 건너뛰기
            # if i in face_keypoints:
            #     continue
                
            if i < len(reference_keypoints):
                ref = reference_keypoints[i]
                cur = keypoints[i]
                
                if cur[0] == 0 or cur[1] == 0:
                    continue
                
                # 두 포인트가 모두 유효한 경우에만 계산
                if ref[0] is not None and cur[0] is not None and not np.isnan(cur[0]) and not np.isnan(cur[1]):
                    dist = np.linalg.norm(np.array(ref) - np.array(cur))
                    
                    # 거리가 임계값을 초과하면 해당 키포인트 강조 표시
                    if dist > threshold_px:
                        # 빨간색 원으로 문제 지점 표시
                        cv2.circle(result_frame, (int(cur[0]), int(cur[1])), 20, (0, 0, 255), 3)
                    else:
                        # 정상 지점은 녹색 원으로 표시
                        cv2.circle(result_frame, (int(cur[0]), int(cur[1])), 20, (0, 255, 0), 3)
        
        feedback = '\n'.join(analyze_pose(keypoints))
        if feedback:
            result_frame = put_korean_text(
                result_frame, 
                feedback, 
                (10, 30), 
                font_size=40, 
                color=(0, 0, 255), 
                thickness=8
            )
        
        return result_frame
        
    except Exception as e:
        print(f"포즈 감지 처리 중 오류: {str(e)}")
        return result_frame

# 카메라 상태에 따른 프레임 처리 함수
async def process_frame_by_status(camera_id, frame, timestamp, status, quality=85, max_width=640, flip=False):
    """카메라 상태에 따라 프레임 처리"""
    global yolo_model, video_recorders, last_pose_detection_times, record_start_time, camera_status
    
    result_frame = frame.copy()
    pose_data = None
    
    # 이전 상태와 현재 상태가 다르고, 현재 상태가 RECORD인 경우 시작 시간 초기화
    prev_status = getattr(process_frame_by_status, 'prev_status', None)
    if prev_status != status and status == CAMERA_STATUS_RECORD:
        record_start_time = time.time()
        print(f"카운트다운 및 녹화 시작: {record_start_time}")
    
    # 현재 상태 저장
    process_frame_by_status.prev_status = status
    
    # 상태별 처리
    if status == CAMERA_STATUS_OFF:
        # 프레임 처리 안함
        return False
    
    if status == CAMERA_STATUS_MASK:
        if check_pose_alignment(result_frame, yolo_model, camera_id=camera_id):
            await set_camera_status(camera_id, CAMERA_STATUS_READY)
        # await set_camera_status(camera_id, CAMERA_STATUS_READY)  # for test
        result_frame = post_process_mask(result_frame, camera_id=camera_id)

    if status == CAMERA_STATUS_READY:
        result_frame = post_process_mask(result_frame, camera_id=camera_id)
    
    if status == CAMERA_STATUS_RECORD:
        # 카운트다운 및 녹화 처리
        now = time.time()
        elapsed = now - record_start_time if record_start_time else 0
        
        if elapsed < 3.0:  # 카운트다운 3초
            remaining = 3 - int(elapsed)
            flip = False
        elif elapsed < 5.9:  # 녹화 2.9초
            remaining = 0 - (elapsed - 3)
            flip = True
        else:
            # 녹화 종료 후 상태 변경
            await set_camera_status(camera_id, CAMERA_STATUS_ON)
            video_path, duration = stop_video_recording(camera_id)
            if video_path and duration > 0:
                asyncio.create_task(upload_video_to_server(camera_id, video_path, duration))
            remaining = -3.0  # 녹화 종료 표시
        
        # 후처리 수행
        result_frame = post_process_record(result_frame, camera_id=camera_id, remaining=remaining)

    
    elif status == CAMERA_STATUS_DETECT:
        # 포즈 감지 수행
        flip = False
        result_frame = post_process_detect(result_frame, camera_id=camera_id)
    
    # 이미지 인코딩 (좌우 반전 처리는 encode_image 함수에서 처리됨)
    image_data = encode_image(result_frame, quality=quality, max_width=max_width, flip=flip)
    
    if image_data:
        # WebSocket을 통해 이미지 전송
        if await send_frame_via_websocket(camera_id, image_data, timestamp, pose_data):
            return True
    
    return False

# 메인 함수
def main():
    """메인 프로그램"""
    global running, api_url, FLIP_HORIZONTAL, POSE_DETECTION_INTERVAL
    
    # 인자 파서 설정
    parser = argparse.ArgumentParser(description="KOMI 웹캠 클라이언트")
    parser.add_argument('--camera', type=str, required=True, 
                        help='카메라 설정 (형식: "카메라ID:인덱스")')
    parser.add_argument('--server', type=str, default="http://localhost:8000",
                        help='서버 URL (기본값: http://localhost:8000)')
    parser.add_argument('--quality', type=int, default=85,
                        help='이미지 압축 품질 (0-100, 기본값: 85)')
    parser.add_argument('--max-width', type=int, default=640,
                        help='이미지 최대 폭 (픽셀, 기본값: 640)')
    parser.add_argument('--fps', type=int, default=15,
                        help='카메라 프레임 레이트 (기본값: 15)')
    parser.add_argument('--flip', action='store_false',
                        help='카메라 이미지 좌우 반전 비활성화')
    parser.add_argument('--pose-interval', type=float, default=1.0,
                        help='포즈 감지 간격 (초, 기본값: 1.0)')
    
    # 인자 파싱
    args = parser.parse_args()
    api_url = args.server
    FLIP_HORIZONTAL = args.flip
    POSE_DETECTION_INTERVAL = args.pose_interval
    
    # 압축 품질 범위 확인
    quality = max(1, min(100, args.quality))  # 1-100 범위로 제한
    
    # 최대 폭 확인
    max_width = max(320, args.max_width)  # 최소 320px
    
    print(f"서버 URL: {api_url}")
    print(f"프레임 레이트: {args.fps}")
    print(f"이미지 품질: {quality}")
    print(f"최대 이미지 폭: {max_width}px")
    print(f"포즈 감지 간격: {POSE_DETECTION_INTERVAL}초")
    if FLIP_HORIZONTAL:
        print("카메라 이미지 좌우 반전: 활성화")
    
    # 종료 시그널 핸들러 등록
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # 카메라 설정 파싱
    cam_parts = args.camera.strip().split(':')
    if len(cam_parts) != 2:
        print("잘못된 카메라 설정 형식. 'ID:인덱스' 형식으로 입력하세요.")
        return
    
    camera_id, camera_index = cam_parts
    try:
        camera_index = int(camera_index)
    except ValueError:
        print(f"잘못된 카메라 인덱스: {camera_index}")
        return
    
    print(f"카메라 설정: {camera_id}:{camera_index}")
    
    # 카메라 스레드 시작
    thread = threading.Thread(
        target=run_camera_thread,
        args=(camera_id, camera_index, args.fps, quality, max_width, FLIP_HORIZONTAL),
        daemon=True
    )
    thread.start()
    print(f"카메라 {camera_id} 스레드 시작됨")
    
    # 메인 스레드 유지
    try:
        while running and thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("키보드 인터럽트 감지")
        running = False
    
    # 종료 처리
    print("종료 중...")
    
    # 스레드가 종료될 때까지 대기
    thread.join(timeout=5.0)
    
    # 남아있는 연결 확인 및 종료
    if camera_id in ws_connections:
        print(f"남은 연결 정리 중: {camera_id}")
        # 메인 스레드에서 비동기 종료 작업 실행
        try:
            asyncio.run(close_camera_connection(camera_id))
        except Exception as e:
            print(f"최종 정리 중 오류: {str(e)}")
    
    print("프로그램 종료")

if __name__ == "__main__":
    main() 