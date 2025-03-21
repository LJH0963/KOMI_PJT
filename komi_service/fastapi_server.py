import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import base64
import cv2
import numpy as np
import time
import asyncio
from datetime import datetime
import os
import argparse
import sys
from typing import Dict, Any, Optional
from collections import deque
import logging

from komi_service.modules.pose_estimation import detect_pose, compare_poses, get_guide_pose

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 명령행 인수 파싱
def parse_args():
    parser = argparse.ArgumentParser(description='KOMI 웹캠 서버 - 실시간 웹캠 스트리밍 및 포즈 감지')
    parser.add_argument('--host', default='0.0.0.0', help='서버 호스트 주소 (기본값: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트 번호 (기본값: 8000)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    
    return parser.parse_args()

# FastAPI 앱 생성
app = FastAPI(
    title="KOMI API",
    description="KOMI 서비스를 위한 API 서버",
    version="0.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 웹소켓 연결 저장
camera_connections = {}  # 카메라 클라이언트 연결
monitor_connections = {}  # 모니터(스트림릿) 클라이언트 연결

# 이미지 버퍼 관리
IMAGE_BUFFER_SIZE = 5  # 각 카메라별 보관할 최대 이미지 수
image_buffers = {}  # 카메라 ID => 이미지 버퍼(deque)
latest_pose_data = {}  # 카메라 ID => 포즈 데이터

# 디버그 모드
DEBUG = False

# 이미지 저장 및 관리
latest_images: Dict[str, Dict[str, Any]] = {}
image_retention_time = 60  # 이미지 보관 시간 (초)

# 클라이언트 연결 정보 저장
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"클라이언트 연결: {client_id} (총 {len(self.active_connections)}개)")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"클라이언트 연결 해제: {client_id} (총 {len(self.active_connections)}개)")
            
    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
            
    async def broadcast(self, message: dict, exclude=None):
        for client_id, connection in self.active_connections.items():
            if exclude and client_id == exclude:
                continue
            await connection.send_json(message)

# 연결 관리자 생성
manager = ConnectionManager()
monitor_manager = ConnectionManager()

# 애플리케이션 이벤트 핸들러
@app.on_event("startup")
async def startup_event():
    logging.info("서버 시작 중")
    os.makedirs("uploads", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "KOMI API 서버에 오신 것을 환영합니다!"}

# 이미지 버퍼에 이미지 저장
def store_image(camera_id: str, image_data: str, timestamp: float = None):
    """카메라별 이미지 버퍼에 이미지 저장"""
    if timestamp is None:
        timestamp = time.time()
        
    if camera_id not in image_buffers:
        image_buffers[camera_id] = deque(maxlen=IMAGE_BUFFER_SIZE)
        
    # 이미지 추가
    image_buffers[camera_id].append({
        "image_data": image_data,
        "timestamp": timestamp
    })
    
    return timestamp

# 웹소켓 엔드포인트 - 카메라 클라이언트용
@app.websocket("/ws/camera")
async def camera_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 카메라 ID 생성
    camera_id = f"camera_{int(time.time())}_{id(websocket)}"
    camera_connections[camera_id] = websocket
    
    # 모든 모니터 클라이언트에 카메라 연결 알림
    for monitor_id, monitor_ws in monitor_connections.items():
        try:
            await monitor_ws.send_text(
                json.dumps({
                    "type": "camera_connected",
                    "camera_id": camera_id,
                    "timestamp": datetime.now().isoformat()
                })
            )
        except Exception as e:
            logger.error(f"모니터 {monitor_id}에 메시지 전송 실패: {str(e)}")
    
    try:
        # 카메라 목록 전송
        for monitor_id, monitor_ws in monitor_connections.items():
            try:
                await monitor_ws.send_text(
                    json.dumps({
                        "type": "cameras_list",
                        "cameras": list(camera_connections.keys())
                    })
                )
            except Exception as e:
                logger.error(f"모니터 {monitor_id}에 메시지 전송 실패: {str(e)}")
        
        # 카메라 클라이언트에 연결 성공 응답
        await websocket.send_text(
            json.dumps({
                "type": "connection_successful",
                "camera_id": camera_id,
                "message": "서버에 연결되었습니다."
            })
        )
        
        # 카메라 메시지 수신 루프
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                
                # 이미지 프레임 처리
                if data.get("type") == "frame" and "image_data" in data:
                    image_data = data["image_data"]
                    timestamp = data.get("timestamp", time.time())
                    
                    # 이미지 버퍼에 저장
                    store_image(camera_id, image_data, timestamp)
                    
                    # 포즈 감지
                    try:
                        pose_results = detect_pose(image_data)
                        latest_pose_data[camera_id] = pose_results
                        
                        # 정확도 계산 (예시)
                        accuracy = 0.8  # 실제로는 compare_poses 함수 등을 사용하여 계산
                        
                        # 모니터 클라이언트에 상태 알림만 전송 (이미지 없이)
                        for monitor_id, monitor_ws in monitor_connections.items():
                            try:
                                await monitor_ws.send_text(
                                    json.dumps({
                                        "type": "status_update",
                                        "camera_id": camera_id,
                                        "has_new_image": True,
                                        "timestamp": timestamp,
                                        "accuracy": accuracy
                                    })
                                )
                            except Exception as e:
                                logger.error(f"모니터 {monitor_id}에 메시지 전송 실패: {str(e)}")
                    except Exception as e:
                        logger.error(f"포즈 감지 오류: {str(e)}")
                
            except json.JSONDecodeError:
                logger.warning(f"잘못된 JSON 메시지: {message[:100]}...")
            except Exception as e:
                logger.error(f"메시지 처리 오류: {str(e)}")
                
    except WebSocketDisconnect:
        # 카메라 연결 해제
        if camera_id in camera_connections:
            del camera_connections[camera_id]
        
        logger.info(f"카메라 연결 해제: {camera_id} (남은 카메라: {len(camera_connections)}개)")
        
        # 모든 모니터 클라이언트에 카메라 연결 해제 알림
        for monitor_id, monitor_ws in monitor_connections.items():
            try:
                await monitor_ws.send_text(
                    json.dumps({
                        "type": "camera_disconnected",
                        "camera_id": camera_id,
                        "timestamp": datetime.now().isoformat()
                    })
                )
            except Exception as e:
                logger.error(f"모니터 {monitor_id}에 메시지 전송 실패: {str(e)}")

# 웹소켓 엔드포인트 - 모니터 클라이언트용
@app.websocket("/ws/monitor")
async def monitor_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 모니터 ID 생성
    monitor_id = f"monitor_{int(time.time())}_{id(websocket)}"
    monitor_connections[monitor_id] = websocket
    
    try:
        # 모니터 클라이언트에 연결 성공 응답
        await websocket.send_text(
            json.dumps({
                "type": "connection_successful",
                "monitor_id": monitor_id,
                "message": "모니터링 서버에 연결되었습니다."
            })
        )
        
        # 현재 연결된 카메라 목록 전송
        await websocket.send_text(
            json.dumps({
                "type": "cameras_list",
                "cameras": list(camera_connections.keys())
            })
        )
        
        # 모니터 메시지 수신 루프
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                
                # 이미지 요청 처리
                if data.get("type") == "request_image" and "camera_id" in data:
                    camera_id = data["camera_id"]
                    
                    # 해당 카메라의 최신 이미지 검색
                    if camera_id in image_buffers and image_buffers[camera_id]:
                        # 최신 이미지 (버퍼의 마지막 항목)
                        latest_image = image_buffers[camera_id][-1]
                        
                        # 포즈 데이터 추가
                        pose_data = latest_pose_data.get(camera_id)
                        
                        # 이미지 데이터와 포즈 데이터 전송
                        await websocket.send_text(
                            json.dumps({
                                "type": "image_data",
                                "camera_id": camera_id,
                                "image_data": latest_image["image_data"],
                                "pose_data": pose_data,
                                "timestamp": latest_image["timestamp"]
                            })
                        )
                    else:
                        # 이미지가 없는 경우
                        await websocket.send_text(
                            json.dumps({
                                "type": "error",
                                "message": f"카메라 {camera_id}의 이미지가 없습니다."
                            })
                        )
            
            except json.JSONDecodeError:
                logger.warning(f"잘못된 JSON 메시지: {message[:100]}...")
            except Exception as e:
                logger.error(f"메시지 처리 오류: {str(e)}")
                
    except WebSocketDisconnect:
        # 모니터 연결 해제
        if monitor_id in monitor_connections:
            del monitor_connections[monitor_id]
        
        logger.info(f"모니터 연결 해제: {monitor_id} (남은 모니터: {len(monitor_connections)}개)")

# 최신 카메라 목록 API
@app.get("/cameras")
async def get_cameras():
    """현재 연결된 카메라 목록 반환"""
    return {
        "cameras": list(camera_connections.keys()),
        "count": len(camera_connections)
    }

# 특정 카메라의 최신 이미지 요청 API
@app.get("/latest_image/{camera_id}")
async def get_latest_image(camera_id: str):
    """특정 카메라의 최신 이미지 반환"""
    if camera_id in image_buffers and image_buffers[camera_id]:
        latest_image = image_buffers[camera_id][-1]
        return {
            "camera_id": camera_id,
            "image_data": latest_image["image_data"],
            "timestamp": latest_image["timestamp"],
            "pose_data": latest_pose_data.get(camera_id)
        }
    else:
        raise HTTPException(status_code=404, detail=f"카메라 {camera_id}의 이미지를 찾을 수 없습니다.")

# 이미지 만료 처리 백그라운드 작업
@app.on_event("startup")
async def start_image_cleanup():
    """정기적으로 오래된 이미지 제거"""
    async def cleanup_task():
        while True:
            await asyncio.sleep(30)  # 30초마다 실행
            current_time = time.time()
            expired_clients = []
            
            for client_id, img_data in latest_images.items():
                if current_time - img_data["timestamp"] > image_retention_time:
                    expired_clients.append(client_id)
            
            for client_id in expired_clients:
                if client_id in latest_images:
                    del latest_images[client_id]
            
            if DEBUG and expired_clients:
                print(f"{len(expired_clients)}개의 만료된 이미지 제거됨")
    
    asyncio.create_task(cleanup_task())

def start_server(host="0.0.0.0", port=8000, debug=False):
    """서버 시작 함수"""
    global DEBUG
    DEBUG = debug
    
    print(f"===== KOMI 웹캠 서버 시작 =====")
    print(f"서버 주소: http://{host}:{port}")
    print(f"WebSocket 엔드포인트: ws://{host}:{port}/ws")
    print(f"모니터링 엔드포인트: ws://{host}:{port}/ws/monitor")
    print(f"디버그 모드: {'활성화' if debug else '비활성화'}")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # 명령행 인수 파싱
    args = parse_args()
    
    # 서버 시작
    start_server(host=args.host, port=args.port, debug=args.debug) 