from fastapi import WebSocket
from typing import Dict, Set, List
from datetime import datetime
import threading
import asyncio
import base64
import numpy as np
import cv2
import json
import time
from contextlib import asynccontextmanager

# 공유 변수 및 데이터 저장소
latest_image_data: Dict[str, str] = {}
latest_timestamps: Dict[str, datetime] = {}
latest_pose_data: Dict[str, dict] = {}  # 카메라 ID별 포즈 데이터 저장
active_connections: Set[WebSocket] = set()
camera_info: Dict[str, dict] = {}
data_lock = threading.Lock()

# 웹소켓 연결 재시도 관련 설정
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 1.0
image_queues = {}
sync_buffer = {}
thread_pool = None

# 앱 상태 관리
app_state = {
    "is_running": True,
    "connected_cameras": 0,
    "active_websockets": 0,
    "start_time": datetime.now(),
    "last_connection_cleanup": datetime.now()
}

# 공통 유틸리티 함수
def update_connection_status(camera_id, status):
    # streamlit_app.py에서 재정의할 함수
    pass

def get_event_loop():
    local = threading.local()
    if not hasattr(local, "loop"):
        local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(local.loop)
    return local.loop

def process_image_in_thread(image_data):
    try:
        if not image_data:
            return None
        img_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

# FastAPI 웹소켓 관련 함수
@asynccontextmanager
async def lifespan_manager(app):
    app_state["start_time"] = datetime.now()
    app_state["last_connection_cleanup"] = datetime.now()
    # cleanup_connections 태스크 시작
    cleanup_task = asyncio.create_task(cleanup_connections())
    yield
    app_state["is_running"] = False
    # 태스크가 완료될 때까지 기다림
    await cleanup_task

async def disconnect_camera(camera_id: str):
    """카메라 연결 해제 및 리소스 정리"""
    with data_lock:
        if camera_id in camera_info:
            # 연결된 WebSocket 종료
            if "websocket" in camera_info[camera_id]:
                try:
                    await camera_info[camera_id]["websocket"].close(code=1000)
                except Exception:
                    pass
            
            # 구독자에게 카메라 연결 해제 알림
            subscribers = camera_info[camera_id].get("subscribers", set()).copy()
            for ws in subscribers:
                try:
                    await ws.send_json({
                        "type": "camera_disconnected",
                        "camera_id": camera_id,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception:
                    pass
            
            # 카메라 정보 삭제
            del camera_info[camera_id]
            
            # 이미지 데이터도 삭제
            if camera_id in latest_image_data:
                del latest_image_data[camera_id]
            if camera_id in latest_timestamps:
                del latest_timestamps[camera_id]
            if camera_id in latest_pose_data:
                del latest_pose_data[camera_id]
            
            return True
    return False

async def cleanup_connections():
    """정기적인 연결 정리 작업"""
    while app_state["is_running"]:
        try:
            now = datetime.now()
            
            # 마지막 정리 후 60초 이상 지났는지 확인
            if (now - app_state["last_connection_cleanup"]).total_seconds() >= 60:
                # 일반 WebSocket 연결 확인
                dead_connections = set()
                for ws in active_connections:
                    try:
                        await ws.send_text("ping")
                    except Exception:
                        dead_connections.add(ws)
                
                # 일반 WebSocket 데드 연결 제거
                for ws in dead_connections:
                    active_connections.discard(ws)
                
                # 카메라 연결 확인 및 정리
                disconnected_cameras = []
                with data_lock:
                    for camera_id, info in list(camera_info.items()):
                        # 마지막 활동 시간이 60초 이상 지난 카메라 확인
                        if "last_seen" in info and (now - info["last_seen"]).total_seconds() >= 60:
                            # 연결 종료 후 정보 삭제
                            disconnected_cameras.append(camera_id)
                
                # 비동기 컨텍스트 밖에서 실행
                for camera_id in disconnected_cameras:
                    await disconnect_camera(camera_id)
                
                # 정리 완료 시간 업데이트
                app_state["last_connection_cleanup"] = now
            
            await asyncio.sleep(10)  # 10초마다 확인
        except Exception:
            await asyncio.sleep(10)  # 오류 발생해도 계속 실행

async def broadcast_image_to_subscribers(camera_id: str, image_data: str, timestamp: datetime, pose_data=None):
    """WebSocket 구독자들에게 이미지 데이터 직접 전송"""
    if camera_id not in camera_info or "subscribers" not in camera_info[camera_id]:
        return
    
    # 메시지 준비
    message = {
        "type": "image",
        "camera_id": camera_id,
        "image_data": image_data,
        "timestamp": timestamp.isoformat()
    }
    
    # 포즈 데이터가 있으면 추가
    if pose_data:
        message["pose_data"] = pose_data
        message["type"] = "image_with_pose"
    
    # 직렬화
    message_str = json.dumps(message)
    
    # 구독자 목록 복사 (비동기 처리 중 변경될 수 있음)
    with data_lock:
        if camera_id in camera_info and "subscribers" in camera_info[camera_id]:
            subscribers = camera_info[camera_id]["subscribers"].copy()
        else:
            return
    
    # 끊어진 연결 추적
    dead_connections = set()
    
    # 모든 구독자에게 전송
    for websocket in subscribers:
        try:
            await websocket.send_text(message_str)
        except Exception:
            dead_connections.add(websocket)
    
    # 끊어진 연결 정리
    if dead_connections:
        with data_lock:
            if camera_id in camera_info and "subscribers" in camera_info[camera_id]:
                camera_info[camera_id]["subscribers"] -= dead_connections

async def notify_clients(camera_id: str, has_pose_data=False):
    """웹소켓 클라이언트에게 이미지 업데이트 알림"""
    if not active_connections:
        return
        
    # 메시지 준비
    message = {
        "type": "image_update",
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat()
    }
    
    if has_pose_data:
        message["type"] = "image_update_with_pose"
        message["has_pose"] = True
    
    message_str = json.dumps(message)
    
    # 연결된 모든 클라이언트에게 알림
    dead_connections = set()
    for websocket in active_connections:
        try:
            await websocket.send_text(message_str)
        except Exception:
            dead_connections.add(websocket)
    
    # 끊어진 연결 정리
    if dead_connections:
        for dead in dead_connections:
            active_connections.discard(dead)
        
        # 상태 업데이트
        app_state["active_websockets"] = len(active_connections)

async def keep_websocket_alive(websocket: WebSocket):
    """WebSocket 연결을 유지하는 함수"""
    ping_interval = 30  # 30초마다 핑 전송
    last_ping_time = time.time()
    last_received_time = time.time()
    max_idle_time = 60  # 60초 동안 응답이 없으면 연결 종료
    
    try:
        while True:
            current_time = time.time()
            
            # 마지막 응답으로부터 너무 오래 경과했는지 확인
            if current_time - last_received_time > max_idle_time:
                return False
            
            # 정기적인 핑 전송
            if current_time - last_ping_time >= ping_interval:
                try:
                    await websocket.send_text("ping")
                    last_ping_time = current_time
                except Exception:
                    return False
            
            # 메시지 수신 시도 (짧은 타임아웃으로 반응성 유지)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                last_received_time = time.time()
                
                # 핑/퐁 처리
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "pong":
                    # 클라이언트에서 보낸 퐁 응답
                    pass
            except asyncio.TimeoutError:
                # 타임아웃은 정상적인 상황, 계속 진행
                pass
            except Exception:
                # 기타 오류 발생 시 연결 종료
                return False
            
            await asyncio.sleep(0.1)
    except Exception:
        return False
    
    return True

async def handle_websocket_connection(websocket: WebSocket):
    """웹소켓 클라이언트 연결 처리"""
    await websocket.accept()
    
    # 연결 목록에 추가
    active_connections.add(websocket)
    
    # 상태 업데이트
    app_state["active_websockets"] = len(active_connections)
    
    # 초기 카메라 목록 전송
    with data_lock:
        active_cameras = list(camera_info.keys())
    
    try:
        # 초기 데이터 전송
        await websocket.send_json({
            "type": "init",
            "cameras": active_cameras,
            "timestamp": datetime.now().isoformat()
        })
        
        # 연결 유지 루프
        if not await keep_websocket_alive(websocket):
            # 연결 유지 실패
            pass
    except Exception:
        # 오류 처리
        pass
    finally:
        # 연결 목록에서 제거
        active_connections.discard(websocket)
        # 상태 업데이트
        app_state["active_websockets"] = len(active_connections)

async def handle_camera_websocket(websocket: WebSocket):
    """웹캠 클라이언트의 WebSocket 연결 처리"""
    await websocket.accept()
    
    camera_id = None
    try:
        # 첫 메시지에서 카메라 ID 확인 또는 생성
        first_message = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        data = json.loads(first_message)
        
        if data.get("type") == "register":
            camera_id = data.get("camera_id")
            
            # 기존 동일 ID 카메라가 있으면 연결 해제
            if camera_id in camera_info:
                await disconnect_camera(camera_id)
        
        # 새 카메라 ID 생성
        if not camera_id:
            camera_id = f"webcam_{len(camera_info) + 1}"
        
        # 카메라 정보 저장
        with data_lock:
            camera_info[camera_id] = {
                "info": data.get("info", {}),
                "last_seen": datetime.now(),
                "websocket": websocket,
                "subscribers": set()  # 구독자 목록 초기화
            }
        
        # 카메라에 ID 전송
        await websocket.send_json({
            "type": "connection_successful",
            "camera_id": camera_id
        })
        
        # 연결 유지 및 프레임 수신 루프
        last_seen = datetime.now()
        last_keepalive = time.time()
        keepalive_interval = 15  # 15초마다 핑 전송
        
        while True:
            try:
                # 짧은 타임아웃으로 메시지 수신 대기
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                
                # 핑/퐁 처리
                if message == "ping":
                    await websocket.send_text("pong")
                    last_seen = datetime.now()
                    continue
                elif message == "pong":
                    last_seen = datetime.now()
                    continue
                
                # JSON 메시지 파싱
                try:
                    data = json.loads(message)
                    
                    # 타입별 메시지 처리
                    msg_type = data.get("type")
                    
                    if msg_type == "frame" or msg_type == "frame_with_pose":
                        # 프레임 저장
                        image_data = data.get("image_data")
                        if image_data:
                            timestamp = datetime.now()
                            last_seen = timestamp
                            
                            # 이미지 저장
                            with data_lock:
                                latest_image_data[camera_id] = image_data
                                latest_timestamps[camera_id] = timestamp
                                
                                # 포즈 데이터가 있으면 저장
                                has_pose_data = False
                                if "pose_data" in data and data["pose_data"]:
                                    latest_pose_data[camera_id] = data["pose_data"]
                                    has_pose_data = True
                                
                                # 카메라 상태 업데이트
                                if camera_id in camera_info:
                                    camera_info[camera_id]["last_seen"] = timestamp
                            
                            # 구독자에게 이미지 직접 전송 (포즈 데이터 포함)
                            pose_data = data.get("pose_data") if has_pose_data else None
                            await broadcast_image_to_subscribers(camera_id, image_data, timestamp, pose_data)
                            
                            # 일반 웹소켓 클라이언트에게 알림
                            await notify_clients(camera_id, has_pose_data)
                    
                    elif msg_type == "disconnect":
                        # 클라이언트에서 종료 요청 - 정상 종료
                        break
                        
                except json.JSONDecodeError:
                    # JSON 파싱 오류는 무시
                    pass
                
                # 정기적인 핑 전송
                current_time = time.time()
                if current_time - last_keepalive >= keepalive_interval:
                    try:
                        await websocket.send_text("ping")
                        last_keepalive = current_time
                    except Exception:
                        # 핑 전송 실패 시 연결 종료
                        break
                    
            except asyncio.TimeoutError:
                # 타임아웃은 정상적인 상황, 핑 체크만 수행
                current_time = time.time()
                if current_time - last_keepalive >= keepalive_interval:
                    try:
                        await websocket.send_text("ping")
                        last_keepalive = current_time
                    except Exception:
                        # 핑 전송 실패 시 연결 종료
                        break
                
                # 장시간 메시지가 없는지 확인 (45초 이상)
                if (datetime.now() - last_seen).total_seconds() > 45:
                    # 너무 오래 메시지가 없으면 연결 종료
                    break
            except Exception:
                # 기타 예외 발생 시 연결 종료
                break
    except Exception:
        pass
    finally:
        # 연결 종료 처리 - 완전히 삭제
        if camera_id:
            await disconnect_camera(camera_id)

async def handle_stream_websocket(websocket: WebSocket, camera_id: str):
    """특정 카메라의 이미지를 WebSocket으로 스트리밍"""
    await websocket.accept()
    
    # 해당 카메라가 존재하는지 확인
    if camera_id not in camera_info:
        await websocket.close(code=1008, reason=f"카메라 ID {camera_id}를 찾을 수 없습니다")
        return
    
    # 해당 카메라의 실시간 스트리밍을 구독하는 클라이언트 등록
    with data_lock:
        if "subscribers" not in camera_info[camera_id]:
            camera_info[camera_id]["subscribers"] = set()
        
        camera_info[camera_id]["subscribers"].add(websocket)
    
    try:
        # 최신 이미지가 있으면 즉시 전송
        with data_lock:
            if camera_id in latest_image_data and camera_id in latest_timestamps:
                image_data = latest_image_data[camera_id]
                timestamp = latest_timestamps[camera_id]
                
                # 포즈 데이터도 있으면 같이 전송
                pose_data = latest_pose_data.get(camera_id)
                
                if image_data:
                    # 이미지 메시지 전송
                    message = {
                        "type": "image" if not pose_data else "image_with_pose",
                        "camera_id": camera_id,
                        "image_data": image_data,
                        "timestamp": timestamp.isoformat()
                    }
                    
                    if pose_data:
                        message["pose_data"] = pose_data
                    
                    await websocket.send_json(message)
        
        # 연결 유지 루프
        if not await keep_websocket_alive(websocket):
            # 연결 유지 실패
            pass
    except Exception:
        # 예외 처리
        pass
    finally:
        # 구독 목록에서 제거
        with data_lock:
            if camera_id in camera_info and "subscribers" in camera_info[camera_id]:
                camera_info[camera_id]["subscribers"].discard(websocket)
