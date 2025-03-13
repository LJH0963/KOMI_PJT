import cv2
import torch
import threading
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import List
from datetime import datetime
from ultralytics import YOLO
import json

app = FastAPI()

# 🔹 스레드 안전성을 위한 Lock 추가
pose_data_lock = threading.Lock()

# 🔹 최신 감지된 포즈 데이터를 저장하는 전역 변수
latest_pose_data = {"status": "waiting", "pose": [], "timestamp": None}

# 📌 YOLO 모델 로드
model = YOLO("./yolov8n-pose.pt")

# 🔹 웹캠 실행 상태 플래그
webcam_running = False

current_index = 0
index_lock = threading.Lock()
mock_data_path = "./II_service/data/json_modified.json"
with open(mock_data_path, "r", encoding="utf-8") as file:
    mock_data = json.load(file)
data_length = len(mock_data)

# 📌 Pydantic 데이터 모델 정의
class Keypoint(BaseModel):
    id: int
    x: int
    y: int
    confidence: float

class PersonPose(BaseModel):
    person_id: int
    keypoints: List[Keypoint]

class PoseResponse(BaseModel):
    status: str
    pose: List[PersonPose]
    timestamp: str
    
class PoseResponseMock(BaseModel):
    status: str
    pose: List[PersonPose]
    timestamp: str
    image_id: str


# 📌 웹캠 프로세스 실행 함수 (스레드 실행)
def capture_webcam():
    global latest_pose_data, webcam_running
    vcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not vcap.isOpened():
        print("웹캠을 열 수 없습니다.")
        webcam_running = False
        return

    webcam_running = True  # 웹캠 실행 상태 업데이트

    while webcam_running:
        ret, frame = vcap.read()
        if not ret:
            continue  # 오류 발생 시 루프 유지

        frame = cv2.flip(frame, 1)
        results = model(frame, verbose=False)  # 🔹 로그 출력 방지

        # 감지된 포즈 데이터 저장
        pose_data = []
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()
            scores = result.keypoints.conf.cpu().numpy()

            keypoints_list = [
                {"id": i, "x": int(kp[0]), "y": int(kp[1]), "confidence": float(score)}
                for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])) if score > 0.5
            ]
            pose_data.append({"person_id": 1, "keypoints": keypoints_list})

        # 🔹 최신 포즈 데이터 갱신
        with pose_data_lock:
            latest_pose_data = {
                "status": "success",
                "pose": pose_data,
                "timestamp": datetime.utcnow().isoformat()
            }

        # 감지된 결과 화면 출력
        cv2.imshow("YOLO Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam_running = False  # 웹캠 실행 중지 상태 업데이트
    vcap.release()
    cv2.destroyAllWindows()

# 📌 FastAPI 라우터 설정
pose_router = APIRouter(prefix="/pose", tags=["pose"])

# 📌 1. 샘플 포즈 데이터 반환 API
@pose_router.get("/sample", response_model=PoseResponse)
async def get_sample_pose():
    """ 샘플 포즈 데이터를 반환하는 API """
    sample_data = PoseResponse(
        status="success",
        pose=[{
            'person_id': 1,
            'keypoints': [
                {'id': 0, 'x': 285, 'y': 119, 'confidence': 0.9949689507484436},
                {'id': 1, 'x': 302, 'y': 102, 'confidence': 0.9841347336769104},
                {'id': 2, 'x': 274, 'y': 104, 'confidence': 0.971875011920929},
                {'id': 3, 'x': 337, 'y': 107, 'confidence': 0.8559675812721252},
                {'id': 4, 'x': 265, 'y': 111, 'confidence': 0.5852421522140503},
                {'id': 5, 'x': 363, 'y': 206, 'confidence': 0.9973533153533936},
                {'id': 6, 'x': 242, 'y': 206, 'confidence': 0.9972413778305054},
                {'id': 7, 'x': 376, 'y': 351, 'confidence': 0.9931977987289429},
                {'id': 8, 'x': 203, 'y': 343, 'confidence': 0.9921365976333618},
                {'id': 9, 'x': 361, 'y': 464, 'confidence': 0.9880317449569702},
                {'id': 10, 'x': 185, 'y': 461, 'confidence': 0.985954225063324},
                {'id': 11, 'x': 334, 'y': 465, 'confidence': 0.9998376369476318},
                {'id': 12, 'x': 241, 'y': 465, 'confidence': 0.9998351335525513},
                {'id': 13, 'x': 385, 'y': 669, 'confidence': 0.9994474053382874},
                {'id': 14, 'x': 209, 'y': 669, 'confidence': 0.9994779229164124},
                {'id': 15, 'x': 432, 'y': 856, 'confidence': 0.9894200563430786},
                {'id': 16, 'x': 183, 'y': 863, 'confidence': 0.9900990128517151}
            ]
        }],
        timestamp=datetime.utcnow().isoformat(),
    )
    return sample_data


# 📌 1-2. Mock 데이터 반환 API (1/30초 간격, 순차적 반환)
@pose_router.get("/mock", response_model=PoseResponseMock)
def get_mock_pose():
    """
    📌 1/30초마다 데이터를 순차적으로 변경하여 반환하는 API
    - 1초에 30개의 데이터가 변경됨 (1프레임 = 1/30초)
    - 총 139개의 데이터가 반복 재생됨
    """
    # 🔹 현재 시간을 밀리초 단위로 변환 후 30프레임으로 나누어 인덱스 계산
    current_time_ms = int(datetime.utcnow().timestamp() * 1000)  # UTC timestamp (밀리초)
    frame_index = (current_time_ms // (1000 // 30)) % data_length  # 🔹 30FPS 기준 인덱스 계산

    # 선택된 데이터 가져오기
    mock_tmp = mock_data[frame_index]
    # print(mock_tmp)
    # 📌 응답 생성
    response = PoseResponseMock(
        status="success",
        timestamp=datetime.utcnow().isoformat(),
        pose=mock_tmp["pose"],  # pose 데이터 포함
        image_id=mock_tmp.get("image_id", "unknown.jpg")  # 🔹 이미지 파일명 포함
    )

    return response
    
# 📌 2. 웹캠 감지 시작 API
@pose_router.post("/start-webcam")
async def start_webcam():
    """ 웹캠 감지를 백그라운드에서 실행하는 API """
    global webcam_running

    if webcam_running:
        return {"message": "웹캠 감지가 이미 실행 중입니다!"}

    thread = threading.Thread(target=capture_webcam, daemon=True)
    thread.start()
    return {"message": "웹캠 감지를 시작합니다. 브라우저를 닫아도 계속 실행됩니다."}

# 📌 3. 웹캠 감지 중지 API
@pose_router.post("/stop-webcam")
async def stop_webcam():
    """ 실행 중인 웹캠 감지를 중지하는 API """
    global webcam_running
    webcam_running = False
    return {"message": "웹캠 감지가 중지되었습니다."}

# 📌 4. 실시간 포즈 데이터 반환 API
@pose_router.get("/live", response_model=PoseResponse)
async def get_live_pose():
    """ 가장 최신의 포즈 데이터를 반환하는 API """
    with pose_data_lock:
        return latest_pose_data

# 📌 FastAPI에 라우터 추가
app.include_router(pose_router)

# 📌 기본 엔드포인트
@app.get("/")
def home():
    return {"message": "안녕하세요! 포즈 감지 API 입니다!"}

# uvicorn II_service.main:app --host 0.0.0.0 --port 8001 --reload
