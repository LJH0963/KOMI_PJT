from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiohttp
import base64
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
import cv2
import numpy as np
import threading
from contextlib import asynccontextmanager

# 저장소: 카메라 ID -> 이미지 데이터
latest_image_data: Dict[str, str] = {}
latest_timestamps: Dict[str, datetime] = {}

# 활성 연결
active_connections: Set[WebSocket] = set()
camera_info: Dict[str, dict] = {}

# 락
data_lock = threading.Lock()

# 상태 관리
app_state = {
    "is_running": True,
    "connected_cameras": 0,
    "active_websockets": 0,
    "start_time": datetime.now()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    app_state["start_time"] = datetime.now()
    yield
    # 종료 시 실행
    app_state["is_running"] = False

app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이미지 디코딩 함수
def decode_image(base64_str):
    """Base64 인코딩된 문자열을 이미지로 디코딩"""
    try:
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"이미지 디코딩 오류: {str(e)}")
        return None

# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    uptime = datetime.now() - app_state["start_time"]
    return {
        "status": "healthy" if app_state["is_running"] else "shutting_down",
        "connected_cameras": len(camera_info),
        "active_websockets": len(active_connections),
        "uptime_seconds": uptime.total_seconds(),
        "uptime_formatted": str(uptime)
    }

# 카메라 등록 엔드포인트
@app.post("/register_camera")
async def register_camera(data: dict):
    """카메라 등록"""
    camera_id = data.get("camera_id")
    info = data.get("info", {})
    
    if not camera_id:
        raise HTTPException(status_code=400, detail="카메라 ID가 필요합니다")
    
    # 카메라 정보 저장
    with data_lock:
        camera_info[camera_id] = {
            "info": info,
            "last_seen": datetime.now()
        }
    
    print(f"카메라 등록됨: {camera_id}")
    return {"status": "success", "camera_id": camera_id}

# 이미지 업로드 엔드포인트
@app.post("/upload_image")
async def upload_image(data: dict, background_tasks: BackgroundTasks):
    """이미지 업로드 및 처리"""
    camera_id = data.get("camera_id")
    image_data = data.get("image_data")
    timestamp_str = data.get("timestamp")
    
    if not camera_id or not image_data:
        raise HTTPException(status_code=400, detail="카메라 ID와 이미지 데이터가 필요합니다")
    
    try:
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
    except ValueError:
        timestamp = datetime.now()
    
    # 이미지 데이터 저장
    with data_lock:
        latest_image_data[camera_id] = image_data
        latest_timestamps[camera_id] = timestamp
        
        # 카메라 상태 업데이트
        if camera_id in camera_info:
            camera_info[camera_id]["last_seen"] = datetime.now()
    
    # 웹소켓 클라이언트에게 알림 (백그라운드)
    background_tasks.add_task(notify_clients, camera_id)
    
    return {"status": "success", "timestamp": timestamp.isoformat()}

# 최신 이미지 조회 엔드포인트
@app.get("/latest_image/{camera_id}")
async def get_latest_image(camera_id: str):
    """카메라의 최신 이미지와 메타데이터 조회"""
    if camera_id not in latest_image_data:
        raise HTTPException(status_code=404, detail="해당 카메라의 이미지가 없습니다")
    
    with data_lock:
        image_data = latest_image_data.get(camera_id)
        timestamp = latest_timestamps.get(camera_id)
    
    return {
        "camera_id": camera_id,
        "image_data": image_data,
        "timestamp": timestamp.isoformat() if timestamp else None
    }

# 이미지 바이너리 조회 엔드포인트
@app.get("/get-image/{camera_id}")
async def get_image(camera_id: str):
    """카메라의 최신 이미지를 바이너리로 제공"""
    if camera_id not in latest_image_data:
        raise HTTPException(status_code=404, detail="해당 카메라의 이미지가 없습니다")
    
    with data_lock:
        image_data = latest_image_data.get(camera_id)
    
    if not image_data:
        raise HTTPException(status_code=404, detail="이미지 데이터가 없습니다")
    
    try:
        # Base64 디코딩
        binary_data = base64.b64decode(image_data)
        return Response(content=binary_data, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 변환 오류: {str(e)}")

# 카메라 목록 조회 엔드포인트
@app.get("/cameras")
async def get_cameras():
    """등록된 카메라 목록 조회"""
    with data_lock:
        # 최근 5분 이내 활동이 있는 카메라만 필터링
        now = datetime.now()
        active_cameras = [
            camera_id for camera_id, info in camera_info.items()
            if (now - info["last_seen"]).total_seconds() < 300  # 5분 = 300초
        ]
    
    return {"cameras": active_cameras, "count": len(active_cameras)}

# 웹소켓 클라이언트 알림 함수
async def notify_clients(camera_id: str):
    """웹소켓 클라이언트에게 이미지 업데이트 알림"""
    if not active_connections:
        return
        
    # 메시지 준비
    message = {
        "type": "image_update",
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat()
    }
    
    message_str = json.dumps(message)
    
    # 연결된 모든 클라이언트에게 알림
    dead_connections = set()
    for websocket in active_connections:
        try:
            await websocket.send_text(message_str)
        except Exception as e:
            print(f"웹소켓 메시지 전송 오류: {str(e)}")
            dead_connections.add(websocket)
    
    # 끊어진 연결 정리
    for dead in dead_connections:
        active_connections.discard(dead)
    
    # 상태 업데이트
    app_state["active_websockets"] = len(active_connections)

# 웹소켓 연결 엔드포인트
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결 처리"""
    await websocket.accept()
    
    # 연결 목록에 추가
    active_connections.add(websocket)
    
    # 상태 업데이트
    app_state["active_websockets"] = len(active_connections)
    
    # 초기 카메라 목록 전송
    now = datetime.now()
    with data_lock:
        active_cameras = [
            camera_id for camera_id, info in camera_info.items()
            if (now - info["last_seen"]).total_seconds() < 300
        ]
    
    try:
        # 초기 데이터 전송
        await websocket.send_json({
            "type": "init",
            "cameras": active_cameras,
            "timestamp": now.isoformat()
        })
        
        # 연결 유지 루프
        while True:
            # 클라이언트로부터 메시지 수신 (핑/퐁 메커니즘)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                
                # 핑에 대한 응답
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # 60초 동안 메시지가 없으면 연결 확인
                try:
                    await websocket.send_text("ping")
                    await asyncio.wait_for(websocket.receive_text(), timeout=5)
                except:
                    # 응답이 없으면 연결 종료로 간주
                    break
            except Exception as e:
                # 기타 오류 발생 시 연결 종료
                print(f"웹소켓 연결 오류: {str(e)}")
                break
    except Exception as e:
        print(f"웹소켓 처리 오류: {str(e)}")
    finally:
        # 연결 목록에서 제거
        active_connections.discard(websocket)
        # 상태 업데이트
        app_state["active_websockets"] = len(active_connections)

# 웹캠 카메라용 WebSocket 엔드포인트
@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """웹캠 클라이언트의 WebSocket 연결 처리"""
    await websocket.accept()
    
    camera_id = None
    try:
        # 첫 메시지에서 카메라 ID 확인 또는 생성
        first_message = await websocket.receive_text()
        data = json.loads(first_message)
        
        if data.get("type") == "register":
            camera_id = data.get("camera_id")
        
        # 새 카메라 ID 생성
        if not camera_id:
            camera_id = f"webcam_{len(camera_info) + 1}"
        
        # 카메라 정보 저장
        with data_lock:
            camera_info[camera_id] = {
                "info": data.get("info", {}),
                "last_seen": datetime.now(),
                "websocket": websocket
            }
        
        # 카메라에 ID 전송
        await websocket.send_json({
            "type": "connection_successful",
            "camera_id": camera_id
        })
        
        print(f"웹캠 연결됨: {camera_id}")
        
        # 연결 유지 및 프레임 수신 루프
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "frame":
                # 프레임 저장
                image_data = data.get("image_data")
                if image_data:
                    timestamp = datetime.now()
                    
                    # 이미지 저장
                    with data_lock:
                        latest_image_data[camera_id] = image_data
                        latest_timestamps[camera_id] = timestamp
                        
                        # 카메라 상태 업데이트
                        if camera_id in camera_info:
                            camera_info[camera_id]["last_seen"] = timestamp
                    
                    # 웹소켓 클라이언트에게 알림
                    await notify_clients(camera_id)
    except Exception as e:
        print(f"웹캠 웹소켓 오류: {str(e)}")
    finally:
        # 연결 종료 처리
        if camera_id and camera_id in camera_info:
            with data_lock:
                if "websocket" in camera_info[camera_id]:
                    del camera_info[camera_id]["websocket"]

# 서버 실행 (직접 실행 시)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 