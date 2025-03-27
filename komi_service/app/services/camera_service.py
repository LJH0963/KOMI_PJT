"""
카메라 관리 서비스
"""

import asyncio
from fastapi import WebSocket
from datetime import datetime
from typing import Set, Optional, Dict, Any

from app.models.camera import CameraModel
from app.utils.websocket_utils import broadcast_message, create_message

class CameraService:
    """카메라 관리 서비스"""
    
    @staticmethod
    async def register_camera(websocket: WebSocket, camera_id: Optional[str], info: Dict[str, Any]) -> str:
        """새 카메라 등록
        
        Args:
            websocket: 웹소켓 연결
            camera_id: 카메라 ID (None이면 자동 생성)
            info: 카메라 정보
            
        Returns:
            등록된 카메라 ID
        """
        # 카메라 ID가 없으면 생성
        if not camera_id:
            active_cameras = CameraModel.get_active_cameras()
            camera_id = f"webcam_{len(active_cameras) + 1}"
        
        # 카메라 정보 등록
        CameraModel.register_camera(camera_id, info, websocket)
        return camera_id
    
    @staticmethod
    async def disconnect_camera(camera_id: str) -> bool:
        """카메라 연결 해제 처리
        
        Args:
            camera_id: 카메라 ID
            
        Returns:
            해제 성공 여부
        """
        # 카메라 정보 가져오기
        camera_info = CameraModel.get_camera_info(camera_id)
        if not camera_info:
            return False
        
        # 연결된 웹소켓 종료
        if "websocket" in camera_info:
            try:
                await camera_info["websocket"].close(code=1000)
            except Exception:
                pass
        
        # 구독자에게 카메라 연결 해제 알림
        subscribers = CameraModel.get_subscribers(camera_id)
        if subscribers:
            message = create_message("camera_disconnected", {"camera_id": camera_id})
            await broadcast_message(subscribers, message)
        
        # 카메라 정보 삭제
        CameraModel.remove_camera(camera_id)
        print(f"카메라 {camera_id} 연결 해제 및 정리 완료")
        return True
    
    @staticmethod
    async def cleanup_inactive_cameras(max_idle_time: int = 60) -> None:
        """비활성 카메라 정리
        
        Args:
            max_idle_time: 최대 허용 비활성 시간 (초)
        """
        now = datetime.now()
        disconnected_cameras = []
        
        # 현재 활성화된 카메라 목록 가져오기
        active_cameras = CameraModel.get_active_cameras()
        
        # 각 카메라의 마지막 활동 시간 확인
        for camera_id in active_cameras:
            camera_info = CameraModel.get_camera_info(camera_id)
            if camera_info and "last_seen" in camera_info:
                if (now - camera_info["last_seen"]).total_seconds() >= max_idle_time:
                    disconnected_cameras.append(camera_id)
        
        # 비활성 카메라 연결 해제
        for camera_id in disconnected_cameras:
            await CameraService.disconnect_camera(camera_id)
    
    @staticmethod
    async def send_image_to_subscribers(camera_id: str, image_data: str, timestamp: datetime) -> None:
        """이미지를 구독자들에게 전송
        
        Args:
            camera_id: 카메라 ID
            image_data: 이미지 데이터 (Base64)
            timestamp: 타임스탬프
        """
        subscribers = CameraModel.get_subscribers(camera_id)
        if not subscribers:
            return
        
        # 메시지 생성
        message = create_message("image", {
            "camera_id": camera_id,
            "image_data": image_data
        })
        
        # 구독자에게 메시지 전송
        dead_connections = await broadcast_message(subscribers, message)
        
        # 연결 끊긴 구독자 정리
        for ws in dead_connections:
            CameraModel.remove_subscriber(camera_id, ws)
    
    @staticmethod
    async def send_pose_to_subscribers(camera_id: str, pose_data: Dict[str, Any], timestamp: datetime) -> None:
        """포즈 데이터를 구독자들에게 전송
        
        Args:
            camera_id: 카메라 ID
            pose_data: 포즈 데이터
            timestamp: 타임스탬프
        """
        subscribers = CameraModel.get_subscribers(camera_id)
        if not subscribers:
            return
        
        # 메시지 생성
        message = create_message("pose_data", {
            "camera_id": camera_id,
            "pose_data": pose_data
        })
        
        # 구독자에게 메시지 전송
        dead_connections = await broadcast_message(subscribers, message)
        
        # 연결 끊긴 구독자 정리
        for ws in dead_connections:
            CameraModel.remove_subscriber(camera_id, ws)
    
    @staticmethod
    async def update_clients(camera_id: str, client_connections: Set[WebSocket]) -> None:
        """클라이언트에 카메라 업데이트 알림
        
        Args:
            camera_id: 카메라 ID
            client_connections: 클라이언트 웹소켓 연결 목록
        """
        if not client_connections:
            return
        
        # 메시지 생성
        message = create_message("image_update", {"camera_id": camera_id})
        
        # 클라이언트에게 메시지 전송
        await broadcast_message(client_connections, message) 