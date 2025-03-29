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
data_lock = threading.RLock()

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
    "last_connection_cleanup": datetime.now(),
    "background_tasks": []
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
    """FastAPI 애플리케이션 수명 주기 관리"""
    app_state["is_running"] = True
    app_state["start_time"] = datetime.now()
    app_state["background_tasks"] = []
    
    yield
    
    # 종료 처리
    app_state["is_running"] = False
    
    # 연결된 모든 웹소켓 종료
    with data_lock:
        for ws in active_connections.copy():
            try:
                await ws.close()
            except Exception:
                pass
        active_connections.clear()
    
    # 백그라운드 태스크 종료
    for task in app_state["background_tasks"]:
        task.cancel()
    
    await asyncio.gather(*app_state["background_tasks"], return_exceptions=True)

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
    """일반 클라이언트 웹소켓 연결 처리"""
    await websocket.accept()
    
    with data_lock:
        active_connections.add(websocket)
    
    try:
        while app_state["is_running"]:
            # 클라이언트에서 전송된 메시지 수신
            msg = await websocket.receive_text()
            
            # 주기적인 업데이트 요청 처리
            if msg == "get_updates":
                with data_lock:
                    # 카메라 정보 반환
                    active_cameras = [
                        {"id": camera_id, "status": info.get("status", "disconnected")}
                        for camera_id, info in camera_info.items()
                        if "websocket" in info
                    ]
                
                await websocket.send_json({
                    "type": "camera_update",
                    "cameras": active_cameras,
                    "timestamp": time.time()
                })
    except Exception as e:
        print(f"웹소켓 오류: {str(e)}")
    finally:
        with data_lock:
            if websocket in active_connections:
                active_connections.remove(websocket)

async def handle_camera_websocket(websocket: WebSocket):
    """카메라 클라이언트 웹소켓 연결 처리"""
    await websocket.accept()
    camera_id = None
    
    try:
        # 카메라 등록 메시지 수신
        data = await websocket.receive_json()
        
        if data.get("type") == "register":
            camera_id = data.get("camera_id")
            info = data.get("info", {})
            
            if not camera_id:
                await websocket.close(code=1000, reason="카메라 ID가 필요합니다")
                return
            
            # 카메라 정보 저장
            with data_lock:
                if camera_id in camera_info and "websocket" in camera_info[camera_id]:
                    old_ws = camera_info[camera_id]["websocket"]
                    try:
                        await old_ws.close(code=1000, reason="다른 연결에 의해 대체됨")
                    except Exception:
                        pass
                
                camera_info[camera_id] = {
                    "websocket": websocket,
                    "connected_at": datetime.now().isoformat(),
                    "info": info,
                    "status": "off"  # 초기 상태는 꺼짐
                }
            
            # 연결 성공 응답
            await websocket.send_json({
                "type": "connection_successful",
                "camera_id": camera_id
            })
            
            # 다른 클라이언트에 카메라 연결 알림
            for ws in active_connections:
                try:
                    await ws.send_json({
                        "type": "camera_connected",
                        "camera_id": camera_id,
                        "info": info
                    })
                except Exception:
                    pass
            
            # 메시지 루프
            while app_state["is_running"]:
                msg = await websocket.receive_json()
                
                # 프레임 처리는 스트림 웹소켓에서 수행하므로 여기서는 다른 메시지 처리
                msg_type = msg.get("type", "")
                
                # 상태 변경 알림 처리
                if msg_type == "status_changed":
                    new_status = msg.get("status")
                    with data_lock:
                        if camera_id in camera_info:
                            camera_info[camera_id]["status"] = new_status
                
                # 녹화 완료 알림 처리
                elif msg_type == "recording_completed":
                    video_path = msg.get("video_path")
                    duration = msg.get("duration", 0)
                    
                    with data_lock:
                        if camera_id in camera_info:
                            camera_info[camera_id]["last_recording"] = {
                                "path": video_path,
                                "duration": duration,
                                "completed_at": datetime.now().isoformat()
                            }
                    
                    # 다른 클라이언트에 녹화 완료 알림
                    for ws in active_connections:
                        try:
                            await ws.send_json({
                                "type": "recording_completed",
                                "camera_id": camera_id,
                                "video_path": video_path,
                                "duration": duration
                            })
                        except Exception:
                            pass
        
    except Exception as e:
        print(f"카메라 웹소켓 오류: {str(e)}")
    finally:
        if camera_id:
            with data_lock:
                if camera_id in camera_info:
                    # 웹소켓 필드만 삭제하고 다른 정보는 유지
                    if "websocket" in camera_info[camera_id]:
                        del camera_info[camera_id]["websocket"]
                    camera_info[camera_id]["disconnected_at"] = datetime.now().isoformat()
                    camera_info[camera_id]["status"] = "disconnected"
            
            # 다른 클라이언트에 카메라 연결 해제 알림
            for ws in active_connections:
                try:
                    await ws.send_json({
                        "type": "camera_disconnected",
                        "camera_id": camera_id
                    })
                except Exception:
                    pass

async def handle_stream_websocket(websocket: WebSocket, camera_id: str):
    """이미지 스트리밍을 위한 웹소켓 연결 처리"""
    # 클라이언트 연결 수락
    await websocket.accept()
    
    # 스트리밍 클라이언트 정보
    streaming_info = {
        "websocket": websocket,
        "camera_id": camera_id,
        "connected_at": datetime.now().isoformat(),
        "last_frame_time": None
    }
    
    # 등록된 카메라인지 확인
    with data_lock:
        if camera_id not in camera_info or "websocket" not in camera_info[camera_id]:
            await websocket.close(code=1000, reason="카메라가 연결되어 있지 않습니다")
            return
        
        # 스트리밍 클라이언트 추가
        if "streaming_clients" not in camera_info[camera_id]:
            camera_info[camera_id]["streaming_clients"] = []
        
        camera_info[camera_id]["streaming_clients"].append(streaming_info)
    
    try:
        # 클라이언트 연결 유지
        while app_state["is_running"]:
            # 메시지 수신 (필요한 경우)
            await asyncio.sleep(1)
    
    except Exception as e:
        print(f"스트림 웹소켓 오류: {str(e)}")
    finally:
        # 스트리밍 클라이언트 목록에서 제거
        with data_lock:
            if (camera_id in camera_info and 
                "streaming_clients" in camera_info[camera_id]):
                
                # 현재 클라이언트 제거
                camera_info[camera_id]["streaming_clients"] = [
                    client for client in camera_info[camera_id]["streaming_clients"]
                    if client["websocket"] != websocket
                ]
