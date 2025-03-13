import torch
import numpy as np
from ultralytics import YOLO
import cv2
from datetime import datetime
from typing import List, Dict

# 📌 1. YOLO-Pose 모델 로드
model = YOLO("yolov8n-pose.pt")  # YOLO-Pose 모델 (경량 버전)

# 📌 2. 이미지 로드
image_path = "tests/KSG/data/sample.jpg"  # 분석할 이미지 경로
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Cannot load image from {image_path}")
    exit()

# 📌 3. YOLO-Pose 모델을 사용하여 포즈 감지
results = model(image)  # YOLO 모델 실행

# 📌 4. 관절(Keypoints) 좌표 추출
pose_data = []  # 전체 인식된 사람들의 데이터를 저장

for person_id, result in enumerate(results):
    keypoints = result.keypoints.xy.cpu().numpy()  # 🟢 GPU → CPU 변환
    scores = result.keypoints.conf.cpu().numpy()  # 🟢 신뢰도 값도 CPU 변환

    # 📌 5. 각 사람별 Keypoints 데이터 정리
    keypoints_list = []
    for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])):  
        if score > 0.5:  # 신뢰도 50% 이상인 경우만 포함
            keypoints_list.append({
                "id": i,  # 관절 ID (COCO 데이터셋 기준)
                "x": int(kp[0]),  # x 좌표
                "y": int(kp[1]),  # y 좌표
                "confidence": float(score)  # 신뢰도 값 (0~1)
            })

    # 📌 6. 사람 단위로 데이터 저장
    pose_data.append({
        "person_id": person_id + 1,  # 감지된 사람 ID
        "keypoints": keypoints_list  # 관절 리스트
    })

# 📌 7. 최종 데이터 구조
pose_response = {
    "status": "success",  # 요청 성공 여부
    "pose": pose_data,  # 감지된 포즈 데이터
    "timestamp": datetime.utcnow().isoformat()  # 타임스탬프
}

# 📌 8. 데이터 출력 (통신 포맷에 맞춰 가공 완료)
print(pose_response)
