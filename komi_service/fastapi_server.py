from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi_websocket import (
    active_connections,
    camera_info,
    data_lock,
    app_state,
    lifespan_manager,
    handle_websocket_connection,
    handle_camera_websocket,
    handle_stream_websocket
)

app = FastAPI(lifespan=lifespan_manager)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    uptime = datetime.now() - app_state["start_time"]
    return {
        "status": "healthy" if app_state["is_running"] else "shutting_down",
        "connected_cameras": len(camera_info),
        "active_websockets": len(active_connections),
        "uptime_seconds": uptime.total_seconds(),
        "uptime_formatted": str(uptime)
    }

# 서버 시간 엔드포인트
@app.get("/server_time")
async def get_server_time():
    """서버의 현재 시간 정보 제공"""
    now = datetime.now()
    return {
        "server_time": now.isoformat(),
        "timestamp": time.time()
    }

# 카메라 목록 조회 엔드포인트
@app.get("/cameras")
async def get_cameras():
    """등록된 카메라 목록 조회"""
    with data_lock:
        # 현재 연결된 카메라만 반환 (WebSocket이 있는 카메라)
        active_cameras = [
            camera_id for camera_id, info in camera_info.items()
            if "websocket" in info
        ]
    
    return {"cameras": active_cameras, "count": len(active_cameras)}

# 웹소켓 연결 엔드포인트
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결 처리"""
    await handle_websocket_connection(websocket)

# 웹캠 카메라용 WebSocket 엔드포인트
@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """웹캠 클라이언트의 WebSocket 연결 처리"""
    await handle_camera_websocket(websocket)

# WebSocket을 통한 이미지 스트리밍 엔드포인트
@app.websocket("/ws/stream/{camera_id}")
async def stream_camera(websocket: WebSocket, camera_id: str):
    """특정 카메라의 이미지를 WebSocket으로 스트리밍"""
    await handle_stream_websocket(websocket, camera_id)

# 서버 실행 (직접 실행 시)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 