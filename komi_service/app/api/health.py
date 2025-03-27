"""
서버 상태 및 헬스 체크 API 라우터
"""

import time
from datetime import datetime
from fastapi import APIRouter

from app.models.camera import camera_info
from app.utils.websocket_utils import active_connections

# 서버 상태 관리
app_state = {
    "is_running": True,
    "start_time": datetime.now(),
    "last_connection_cleanup": datetime.now()
}

router = APIRouter(tags=["health"])

@router.get("/health")
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

@router.get("/server_time")
async def get_server_time():
    """서버의 현재 시간 정보 제공"""
    now = datetime.now()
    return {
        "server_time": now.isoformat(),
        "timestamp": time.time()
    } 