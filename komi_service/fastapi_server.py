import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import json
import base64
import cv2
import numpy as np
import time
import asyncio
from datetime import datetime
import os
import logging
from typing import Dict, Any, Optional, List, Set
from collections import deque
from contextlib import asynccontextmanager

from komi_service.modules.pose_estimation import detect_pose, compare_poses, get_guide_pose

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 이미지 버퍼 관리 (최적화를 위해 컬렉션 재정의)
IMAGE_BUFFER_SIZE = 5
image_buffers: Dict[str, deque] = {}
latest_pose_data: Dict[str, Any] = {}
processing_tasks: Set[asyncio.Task] = set()

# lifespan 이벤트 컨텍스트 매니저
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    logging.info("서버 시작 중")
    os.makedirs("uploads", exist_ok=True)
    # 이미지 정리 백그라운드 작업 시작
    cleanup_task = asyncio.create_task(cleanup_old_images())
    
    yield  # 앱 실행
    
    # 종료 시 실행
    # 실행 중인 모든 작업 취소
    for task in processing_tasks:
        if not task.done():
            task.cancel()
    
    # 정리 작업 취소
    cleanup_task.cancel()
    
    # 작업이 완료될 때까지 대기
    if processing_tasks:
        await asyncio.gather(*processing_tasks, return_exceptions=True)
    
    logging.info("서버 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="KOMI API",
    description="KOMI 서비스를 위한 API 서버",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 클라이언트 연결 정보 저장
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = websocket
        logger.info(f"클라이언트 연결: {client_id} (총 {len(self.active_connections)}개)")
        
    async def disconnect(self, client_id: str):
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                logger.info(f"클라이언트 연결 해제: {client_id} (총 {len(self.active_connections)}개)")
            
    async def send_message(self, message: dict, client_id: str):
        async with self._lock:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    logger.error(f"메시지 전송 오류({client_id}): {str(e)}")
                    await self.disconnect(client_id)
            
    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        # 브로드캐스트를 위한 비동기 작업 생성
        tasks = []
        async with self._lock:
            for client_id, connection in self.active_connections.items():
                if exclude and client_id == exclude:
                    continue
                tasks.append(self._send_to_client(connection, message, client_id))
                
        # 비동기 병렬 처리
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_client(self, websocket: WebSocket, message: dict, client_id: str):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"브로드캐스트 오류({client_id}): {str(e)}")
            await self.disconnect(client_id)

# 카메라와 모니터 연결 관리자 생성
camera_manager = ConnectionManager()
monitor_manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "KOMI API 서버에 오신 것을 환영합니다!"}

# 이미지 처리 작업을 위한 비동기 함수
async def process_image(camera_id: str, image_data: str, timestamp: float):
    """이미지 처리 및 포즈 감지를 비동기적으로 수행"""
    try:
        # 이미지 버퍼에 저장
        if camera_id not in image_buffers:
            image_buffers[camera_id] = deque(maxlen=IMAGE_BUFFER_SIZE)
            
        image_buffers[camera_id].append({
            "image_data": image_data,
            "timestamp": timestamp
        })
        
        # 포즈 감지 (CPU 집약적 작업을 별도 스레드로 분리)
        pose_results = await asyncio.to_thread(detect_pose, image_data)
        latest_pose_data[camera_id] = pose_results
        
        # 모니터에 상태 업데이트 알림
        await monitor_manager.broadcast({
            "type": "status_update",
            "camera_id": camera_id,
            "has_new_image": True,
            "timestamp": timestamp,
            "pose_data": pose_results
        })
        
    except Exception as e:
        logger.error(f"이미지 처리 오류: {str(e)}")

# 웹소켓 엔드포인트 - 카메라 클라이언트용
@app.websocket("/ws/camera")
async def camera_websocket_endpoint(websocket: WebSocket):
    # 카메라 ID 생성
    camera_id = f"camera_{int(time.time())}_{id(websocket)}"
    await camera_manager.connect(websocket, camera_id)
    
    # 모든 모니터에 카메라 연결 알림
    await monitor_manager.broadcast({
        "type": "camera_connected",
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        # 카메라 목록 업데이트
        await monitor_manager.broadcast({
            "type": "cameras_list",
            "cameras": list(camera_manager.active_connections.keys())
        })
        
        # 카메라 클라이언트에 연결 성공 응답
        await websocket.send_json({
            "type": "connection_successful",
            "camera_id": camera_id,
            "message": "서버에 연결되었습니다."
        })
        
        # 카메라 메시지 수신 루프
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                
                # 이미지 프레임 처리
                if data.get("type") == "frame" and "image_data" in data:
                    image_data = data["image_data"]
                    timestamp = data.get("timestamp", time.time())
                    
                    # 비동기 이미지 처리 작업 생성
                    task = asyncio.create_task(process_image(camera_id, image_data, timestamp))
                    processing_tasks.add(task)
                    # 완료 시 작업 세트에서 제거
                    task.add_done_callback(lambda t: processing_tasks.discard(t))
                
            except json.JSONDecodeError:
                logger.warning(f"잘못된 JSON 메시지: {message[:100]}...")
            except Exception as e:
                logger.error(f"메시지 처리 오류: {str(e)}")
                
    except WebSocketDisconnect:
        # 카메라 연결 해제
        await camera_manager.disconnect(camera_id)
        
        # 모든 모니터에 카메라 연결 해제 알림
        await monitor_manager.broadcast({
            "type": "camera_disconnected",
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat()
        })

# 웹소켓 엔드포인트 - 모니터 클라이언트용
@app.websocket("/ws/monitor")
async def monitor_websocket_endpoint(websocket: WebSocket):
    # 모니터 ID 생성
    monitor_id = f"monitor_{int(time.time())}_{id(websocket)}"
    await monitor_manager.connect(websocket, monitor_id)
    
    try:
        # 모니터 클라이언트에 연결 성공 응답
        await websocket.send_json({
            "type": "connection_successful",
            "monitor_id": monitor_id,
            "message": "모니터링 서버에 연결되었습니다."
        })
        
        # 현재 연결된 카메라 목록 전송
        await websocket.send_json({
            "type": "cameras_list",
            "cameras": list(camera_manager.active_connections.keys())
        })
        
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
                        await websocket.send_json({
                            "type": "image_data",
                            "camera_id": camera_id,
                            "image_data": latest_image["image_data"],
                            "pose_data": pose_data,
                            "timestamp": latest_image["timestamp"]
                        })
                    else:
                        # 이미지가 없는 경우
                        await websocket.send_json({
                            "type": "error",
                            "message": f"카메라 {camera_id}의 이미지가 없습니다."
                        })
            
            except json.JSONDecodeError:
                logger.warning(f"잘못된 JSON 메시지: {message[:100]}...")
            except Exception as e:
                logger.error(f"메시지 처리 오류: {str(e)}")
                
    except WebSocketDisconnect:
        # 모니터 연결 해제
        await monitor_manager.disconnect(monitor_id)

# 최신 카메라 목록 API
@app.get("/cameras")
async def get_cameras():
    """현재 연결된 카메라 목록 반환"""
    return {
        "cameras": list(camera_manager.active_connections.keys()),
        "count": len(camera_manager.active_connections)
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

# 특정 카메라의 이미지를 직접 바이너리로 반환하는 API
@app.get("/get-image/{camera_id}")
async def get_image(camera_id: str, background_tasks: BackgroundTasks):
    """특정 카메라의 최신 이미지를 바이너리로 직접 반환 (비동기 처리)"""
    # 버퍼에서 이미지 찾기
    if camera_id in image_buffers and image_buffers[camera_id]:
        try:
            # 최신 이미지 가져오기
            latest_image = image_buffers[camera_id][-1]
            image_data = latest_image["image_data"]
            
            # 이미지 처리는 백그라운드 작업으로 실행
            image_bytes = await asyncio.to_thread(process_image_bytes, image_data)
            return Response(content=image_bytes, media_type="image/jpeg")
        except Exception as e:
            logger.error(f"이미지 처리 오류: {str(e)}")
            raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류: {str(e)}")
    
    # 이미지를 찾을 수 없는 경우
    raise HTTPException(status_code=404, detail=f"카메라 {camera_id}의 이미지를 찾을 수 없습니다.")

# 이미지 처리 함수 (CPU 집약적 작업을 위한 함수)
def process_image_bytes(base64_image: str) -> bytes:
    """Base64 인코딩된 이미지를 JPEG 바이트로 변환"""
    img_bytes = base64.b64decode(base64_image)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    _, img_encoded = cv2.imencode(".jpg", frame)
    return img_encoded.tobytes()

# 오래된 이미지 정리 비동기 작업
async def cleanup_old_images():
    """주기적으로 오래된 이미지 버퍼 정리"""
    while True:
        await asyncio.sleep(30)  # 30초마다 실행
        
        current_time = time.time()
        retention_limit = 60  # 60초 이상 된 이미지 제거
        
        # 각 카메라별 이미지 버퍼 확인
        for camera_id in list(image_buffers.keys()):
            if camera_id not in camera_manager.active_connections:
                # 연결이 끊긴 카메라의 버퍼 정리
                del image_buffers[camera_id]
                if camera_id in latest_pose_data:
                    del latest_pose_data[camera_id]
                logger.info(f"연결 해제된 카메라 {camera_id}의 버퍼 정리")

def start_server(host="0.0.0.0", port=8000, debug=False):
    """서버 시작 함수"""
    print(f"===== KOMI 웹캠 서버 시작 =====")
    print(f"서버 주소: http://{host}:{port}")
    print(f"WebSocket 엔드포인트: ws://{host}:{port}/ws/camera")
    print(f"모니터링 엔드포인트: ws://{host}:{port}/ws/monitor")
    print(f"디버그 모드: {'활성화' if debug else '비활성화'}")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='KOMI 웹캠 서버 - 실시간 웹캠 스트리밍 및 포즈 감지')
    parser.add_argument('--host', default='0.0.0.0', help='서버 호스트 주소 (기본값: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트 번호 (기본값: 8000)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    
    args = parser.parse_args()
    
    # 서버 시작
    start_server(host=args.host, port=args.port, debug=args.debug) 