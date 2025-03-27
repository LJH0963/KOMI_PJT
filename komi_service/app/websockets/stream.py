"""
이미지 스트리밍 웹소켓 처리
"""

from fastapi import WebSocket, APIRouter
from typing import Dict, Any

from app.models.camera import CameraModel
from app.utils.websocket_utils import keep_websocket_alive
from app.config import PING_INTERVAL, MAX_IDLE_TIME
from app.websockets.manager import websocket_manager

router = APIRouter()

@router.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """일반 웹소켓 연결 처리 - 카메라 알림 수신"""
    await websocket_manager.connect(websocket)
    
    # 초기 카메라 목록 전송
    active_cameras = CameraModel.get_active_cameras()
    
    try:
        # 초기 데이터 전송
        await websocket.send_json({
            "type": "init",
            "cameras": active_cameras,
            "timestamp": CameraModel.update_camera_timestamp,
        })
        
        # 연결 유지 루프
        await keep_websocket_alive(
            websocket, 
            ping_interval=PING_INTERVAL, 
            max_idle_time=MAX_IDLE_TIME
        )
    except Exception:
        # 오류 처리 - 조용히 진행
        pass
    finally:
        # 연결 목록에서 제거
        websocket_manager.disconnect(websocket)

@router.websocket("/ws/stream/{camera_id}")
async def stream_camera(websocket: WebSocket, camera_id: str):
    """특정 카메라의 이미지를 WebSocket으로 스트리밍"""
    await websocket.accept()
    
    # 해당 카메라가 존재하는지 확인
    if camera_id not in CameraModel.get_active_cameras():
        await websocket.close(code=1008, reason=f"카메라 ID {camera_id}를 찾을 수 없습니다")
        return
    
    # 해당 카메라의 실시간 스트리밍을 구독하는 클라이언트 등록
    CameraModel.add_subscriber(camera_id, websocket)
    
    try:
        # 최신 이미지가 있으면 즉시 전송
        image_data, timestamp = CameraModel.get_latest_image(camera_id)
        if image_data and timestamp:
            # 이미지 메시지 전송
            await websocket.send_json({
                "type": "image",
                "camera_id": camera_id,
                "image_data": image_data,
                "timestamp": timestamp.isoformat()
            })
        
        # 연결 유지 루프
        await keep_websocket_alive(
            websocket, 
            ping_interval=PING_INTERVAL, 
            max_idle_time=MAX_IDLE_TIME
        )
    except Exception:
        # 예외 처리 - 조용히 진행
        pass
    finally:
        # 구독 목록에서 제거
        CameraModel.remove_subscriber(camera_id, websocket) 