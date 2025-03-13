from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import shutil
import os

# FastAPI 앱 생성
app = FastAPI()

# 📌 Pose 관련 엔드포인트 라우터
pose_router = APIRouter(prefix="/pose", tags=["pose"])

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

# 📌 1. GET 요청 시 샘플 데이터 반환
@pose_router.get("/sample", response_model=PoseResponse)
async def get_sample_pose():
    """
    샘플 포즈 데이터를 반환하는 API
    """
    sample_data = PoseResponse(
        status="success",
        pose=[
            {'person_id': 1,
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
            }
        ],
        timestamp=datetime.utcnow().isoformat(),
    )
    return sample_data

# 📌 2. 이미지 업로드 후 포즈 감지 요청
@pose_router.post("/detect", response_model=PoseResponse)
async def detect_pose(image: UploadFile = File(...)):
    """
    업로드된 이미지를 기반으로 포즈 감지 요청을 수행
    """
    try:
        # 이미지 파일 저장 경로 설정
        image_dir = "uploaded_images"
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, image.filename)

        # 업로드된 파일 저장
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 📌 YOLO-Pose 관련 처리 (외부 모듈 사용 예정)
        # `PoseService.process_pose(image_path)` 형태로 호출 가능하도록 설계
        pose_data = []  # YOLO-Pose 모듈에서 받아올 데이터 구조

        # 📌 응답 데이터 생성
        response = PoseResponse(
            status="success",
            pose=pose_data,
            timestamp=datetime.utcnow().isoformat()
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# FastAPI에 라우터 추가
app.include_router(pose_router)


# 📌 기본 엔드포인트
@app.get("/")
def home():
    return {"message": "안녕하세요! 포즈 감지 API 입니다!"}


# uvicorn service.main:app --host 0.0.0.0 --port 8001 --reload