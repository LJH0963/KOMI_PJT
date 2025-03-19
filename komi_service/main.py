import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter, UploadFile, File
from datetime import datetime
import asyncio

from komi_service.modules.pose_estimation import process_pose
from komi_service.modules.websocket_manager import ws_manager

app = FastAPI()

# 📌 FastAPI 라우터 설정
pose_router = APIRouter(prefix="/pose", tags=["pose"])

# 📌 1. 웹캠 장비에서 이미지 업로드 API
@pose_router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    📌 웹캠 장비에서 이미지를 업로드하는 API
    - 웹캠 장비가 주기적으로 이미지를 서버로 전송
    - YOLO Pose 모델을 통해 포즈 감지 수행
    - 감지된 데이터를 웹소켓을 통해 실시간으로 클라이언트에게 전송
    """
    try:
        # 🔹 이미지 읽기 및 OpenCV 변환
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 🔹 YOLO Pose 감지 실행
        response_data = process_pose(frame)

        # 🔹 웹소켓을 통해 실시간 데이터 전송
        await ws_manager.send_json(response_data)

        return {"message": "이미지 처리 완료", "data": response_data}

    except Exception as e:
        return {"error": f"이미지 처리 중 오류 발생: {str(e)}"}

# 📌 2. 웹소켓 연결 엔드포인트
@pose_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    📌 웹소켓 연결을 통해 실시간 데이터 전송
    - 클라이언트(Streamlit)가 연결하면, 지속적으로 포즈 데이터를 수신
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # 서버에서 주기적으로 클라이언트 확인
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# 📌 FastAPI에 라우터 추가
app.include_router(pose_router)

@app.get("/")
def home():
    return {"message": "포즈 감지 API"}

# uvicorn komi_service.main:app --port 8001 --reload
# uvicorn komi_service.main:app --host 0.0.0.0 --port 8001