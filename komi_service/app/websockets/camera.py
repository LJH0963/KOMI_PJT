"""
카메라 웹소켓 연결 처리
"""

import asyncio
import json
from fastapi import WebSocket, APIRouter
from typing import Dict, Any, Optional

from app.services.camera_service import CameraService
from app.models.camera import CameraModel
from app.config import CAMERA_PING_INTERVAL
from app.websockets.manager import websocket_manager

router = APIRouter()

@router.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
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
            if camera_id:
                await CameraService.disconnect_camera(camera_id)
        
        # 카메라 등록
        camera_id = await CameraService.register_camera(
            websocket, 
            camera_id, 
            data.get("info", {})
        )
        
        # 카메라에 ID 전송
        await websocket.send_json({
            "type": "connection_successful",
            "camera_id": camera_id
        })
        
        print(f"웹캠 연결됨: {camera_id}")
        
        # 연결 유지 및 프레임 수신 루프
        last_seen = CameraModel.get_camera_info(camera_id)["last_seen"]
        last_keepalive = asyncio.get_event_loop().time()
        
        while True:
            try:
                # 짧은 타임아웃으로 메시지 수신 대기
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                
                # 핑/퐁 처리
                if message == "ping":
                    await websocket.send_text("pong")
                    CameraModel.update_camera_timestamp(camera_id)
                    continue
                elif message == "pong":
                    CameraModel.update_camera_timestamp(camera_id)
                    continue
                
                # JSON 메시지 파싱
                try:
                    data = json.loads(message)
                    
                    # 타입별 메시지 처리
                    msg_type = data.get("type")
                    
                    if msg_type == "frame":
                        # 프레임 저장
                        image_data = data.get("image_data")
                        if image_data:
                            # 이미지 저장
                            timestamp = CameraModel.update_image_data(camera_id, image_data)
                            
                            # 구독자에게 이미지 직접 전송
                            await CameraService.send_image_to_subscribers(camera_id, image_data, timestamp)
                            
                            # 일반 웹소켓 클라이언트에게 알림
                            await CameraService.update_clients(camera_id, websocket_manager.active_connections)
                    
                    elif msg_type == "pose_data":
                        # 포즈 데이터 처리
                        pose_data = data.get("pose_data")
                        image_data = data.get("image_data")
                        
                        if pose_data:
                            # 포즈 데이터와 이미지를 함께 저장하는 경우
                            if image_data:
                                timestamp = CameraModel.update_image_data(camera_id, image_data)
                                
                                # 구독자에게 이미지 직접 전송
                                await CameraService.send_image_to_subscribers(camera_id, image_data, timestamp)
                            
                            # 구독자에게 포즈 데이터 전송
                            await CameraService.send_pose_to_subscribers(camera_id, pose_data, timestamp)
                            
                            # 일반 웹소켓 클라이언트에게 알림
                            await CameraService.update_clients(camera_id, websocket_manager.active_connections)
                    
                    elif msg_type == "disconnect":
                        # 클라이언트에서 종료 요청 - 정상 종료
                        print(f"카메라 {camera_id}에서 연결 종료 요청을 받음")
                        break
                        
                except json.JSONDecodeError:
                    # JSON 파싱 오류는 무시
                    pass
                
                # 정기적인 핑 전송
                current_time = asyncio.get_event_loop().time()
                if current_time - last_keepalive >= CAMERA_PING_INTERVAL:
                    try:
                        await websocket.send_text("ping")
                        last_keepalive = current_time
                    except Exception:
                        # 핑 전송 실패 시 연결 종료
                        break
                    
            except asyncio.TimeoutError:
                # 타임아웃은 정상적인 상황, 핑 체크만 수행
                current_time = asyncio.get_event_loop().time()
                if current_time - last_keepalive >= CAMERA_PING_INTERVAL:
                    try:
                        await websocket.send_text("ping")
                        last_keepalive = current_time
                    except Exception:
                        # 핑 전송 실패 시 연결 종료
                        break
                
                # 장시간 메시지가 없는지 확인 (45초 이상)
                camera_info = CameraModel.get_camera_info(camera_id)
                if camera_info and (camera_info["last_seen"] - last_seen).total_seconds() > 45:
                    # 너무 오래 메시지가 없으면 연결 종료
                    print(f"카메라 {camera_id} 45초 동안 활동 없음, 연결 종료")
                    break
            except Exception as e:
                # 기타 예외 발생 시 연결 종료
                print(f"카메라 {camera_id} 처리 오류: {str(e)}")
                break
    except Exception as e:
        print(f"웹캠 웹소켓 오류: {str(e)}")
    finally:
        # 연결 종료 처리 - 완전히 삭제
        if camera_id:
            print(f"카메라 {camera_id} 연결 종료 처리 중...")
            await CameraService.disconnect_camera(camera_id) 