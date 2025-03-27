"""
카메라 모델 및 상태 관리
"""

from typing import Dict, Set, Optional
from datetime import datetime
from fastapi import WebSocket
import threading

# 카메라 정보 저장
camera_info: Dict[str, dict] = {}

# 카메라 이미지 데이터 저장소
latest_image_data: Dict[str, str] = {}
latest_timestamps: Dict[str, datetime] = {}

# 데이터 접근 락
data_lock = threading.Lock()

class CameraModel:
    """카메라 관련 데이터 모델"""
    
    @staticmethod
    def get_active_cameras() -> list:
        """현재 활성화된 카메라 목록 반환"""
        with data_lock:
            # 현재 연결된 카메라만 반환 (WebSocket이 있는 카메라)
            active_cameras = [
                camera_id for camera_id, info in camera_info.items()
                if "websocket" in info
            ]
        return active_cameras
    
    @staticmethod
    def get_camera_info(camera_id: str) -> Optional[dict]:
        """특정 카메라 정보 반환"""
        with data_lock:
            if camera_id in camera_info:
                return camera_info[camera_id].copy()
        return None
    
    @staticmethod
    def register_camera(camera_id: str, info: dict, websocket: WebSocket) -> None:
        """새 카메라 등록"""
        with data_lock:
            camera_info[camera_id] = {
                "info": info,
                "last_seen": datetime.now(),
                "websocket": websocket,
                "subscribers": set()  # 구독자 목록 초기화
            }
    
    @staticmethod
    def update_camera_timestamp(camera_id: str) -> None:
        """카메라 마지막 활동 시간 업데이트"""
        with data_lock:
            if camera_id in camera_info:
                camera_info[camera_id]["last_seen"] = datetime.now()
    
    @staticmethod
    def update_image_data(camera_id: str, image_data: str) -> datetime:
        """이미지 데이터 업데이트"""
        timestamp = datetime.now()
        with data_lock:
            latest_image_data[camera_id] = image_data
            latest_timestamps[camera_id] = timestamp
            
            # 카메라 상태 업데이트
            if camera_id in camera_info:
                camera_info[camera_id]["last_seen"] = timestamp
        
        return timestamp
    
    @staticmethod
    def get_latest_image(camera_id: str) -> tuple:
        """최신 이미지 데이터 및 타임스탬프 반환"""
        with data_lock:
            if camera_id in latest_image_data and camera_id in latest_timestamps:
                return latest_image_data[camera_id], latest_timestamps[camera_id]
            return None, None
    
    @staticmethod
    def add_subscriber(camera_id: str, websocket: WebSocket) -> bool:
        """카메라 구독자 추가"""
        with data_lock:
            if camera_id in camera_info:
                if "subscribers" not in camera_info[camera_id]:
                    camera_info[camera_id]["subscribers"] = set()
                
                camera_info[camera_id]["subscribers"].add(websocket)
                return True
            return False
    
    @staticmethod
    def remove_subscriber(camera_id: str, websocket: WebSocket) -> None:
        """카메라 구독자 제거"""
        with data_lock:
            if camera_id in camera_info and "subscribers" in camera_info[camera_id]:
                camera_info[camera_id]["subscribers"].discard(websocket)
    
    @staticmethod
    def get_subscribers(camera_id: str) -> Set[WebSocket]:
        """카메라 구독자 목록 반환"""
        with data_lock:
            if camera_id in camera_info and "subscribers" in camera_info[camera_id]:
                return camera_info[camera_id]["subscribers"].copy()
            return set()
    
    @staticmethod
    def remove_camera(camera_id: str) -> None:
        """카메라 정보 삭제"""
        with data_lock:
            if camera_id in camera_info:
                del camera_info[camera_id]
                
            # 이미지 데이터도 삭제
            if camera_id in latest_image_data:
                del latest_image_data[camera_id]
            if camera_id in latest_timestamps:
                del latest_timestamps[camera_id] 