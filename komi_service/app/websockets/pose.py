"""
포즈 데이터 웹소켓 처리
"""

from fastapi import WebSocket, APIRouter

from app.models.camera import CameraModel
from app.utils.websocket_utils import keep_websocket_alive
from app.config import PING_INTERVAL, MAX_IDLE_TIME

router = APIRouter()

@router.websocket("/ws/pose/{camera_id}")
async def stream_pose(websocket: WebSocket, camera_id: str):
    """특정 카메라의 포즈 데이터를 WebSocket으로 스트리밍"""
    await websocket.accept()
    
    # 해당 카메라가 존재하는지 확인
    if camera_id not in CameraModel.get_active_cameras():
        await websocket.close(code=1008, reason=f"카메라 ID {camera_id}를 찾을 수 없습니다")
        return
    
    # 해당 카메라의 실시간 스트리밍을 구독하는 클라이언트 등록
    CameraModel.add_subscriber(camera_id, websocket)
    
    try:
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