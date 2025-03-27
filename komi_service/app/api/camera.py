"""
카메라 관련 API 라우터
"""

from fastapi import APIRouter, HTTPException

from app.models.camera import CameraModel

router = APIRouter(prefix="/cameras", tags=["cameras"])

@router.get("")
async def get_cameras():
    """등록된 카메라 목록 조회"""
    active_cameras = CameraModel.get_active_cameras()
    return {"cameras": active_cameras, "count": len(active_cameras)}

@router.get("/{camera_id}")
async def get_camera_info(camera_id: str):
    """카메라 정보 조회"""
    camera_info = CameraModel.get_camera_info(camera_id)
    if not camera_info:
        raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다")
    
    # 민감한 정보 (웹소켓 연결, 구독자 목록 등) 제거
    safe_info = {
        "id": camera_id,
        "info": camera_info.get("info", {}),
        "last_seen": camera_info.get("last_seen"),
        "subscribers_count": len(camera_info.get("subscribers", set()))
    }
    
    return safe_info 