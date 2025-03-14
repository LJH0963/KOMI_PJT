import torch
import numpy as np
import json
import os
import cv2
from datetime import datetime
from tqdm import tqdm  # 진행 상황 표시용
import glob
from ultralytics import YOLO

# 📌 1. YOLO-Pose 모델 로드
model = YOLO("yolov8n-pose.pt")  # YOLO-Pose 경량 모델

# 📌 2. 이미지가 저장된 디렉토리 경로 설정
image_dir = "data/solo_dance/"
output_json = "data/solo_dance_pose_data.json"  # 🔹 JSON 저장 경로

# 📌 3. 저장할 데이터를 담을 리스트
pose_data_list = []

# 📌 4. 디렉토리 내 이미지 파일 리스트 가져오기
image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

# 📌 5. 이미지별로 포즈 감지 수행
for image_name in tqdm(image_files, desc="Processing images"):
    image_path = image_name
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Cannot load image {image_name}")
        continue

    image = cv2.resize(image, (640, 480))  # 이미지 크기 조정

    # YOLO-Pose 모델 실행
    results = model(image)

    # 🔹 현재 이미지의 포즈 데이터를 저장할 리스트
    image_pose_data = {"pose": []}

    for person_id, result in enumerate(results):
        if result.keypoints is None or result.keypoints.xy is None:
            print(f"⚠ 경고: 포즈를 감지하지 못함 - {image_name}")
            continue  # 다음 이미지 처리

        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints.xy is not None else None
        scores = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

        if keypoints is None or scores is None:
            print(f"⚠ 경고: 키포인트 데이터가 존재하지 않음 - {image_name}")
            continue

        # 🔹 개별 사람의 포즈 데이터 저장
        person_pose = {
            "person_id": person_id + 1,
            "keypoints": []
        }

        for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])):
            if score > 0.5:  # 신뢰도 50% 이상인 경우만 저장
                person_pose["keypoints"].append({
                    "id": i,
                    "x": int(kp[0]),
                    "y": int(kp[1]),
                    "confidence": float(score)
                })

        # 사람이 감지된 경우만 저장
        if person_pose["keypoints"]:
            image_pose_data["pose"].append(person_pose)

    # 🔹 이미지별 데이터 저장
    if image_pose_data["pose"]:  # 감지된 데이터가 있을 경우만 추가
        pose_data_list.append(image_pose_data)

# 📌 7. JSON 파일로 저장
if pose_data_list:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pose_data_list, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 저장 완료: {output_json}")
else:
    print("⚠ 경고: 포즈 데이터가 감지되지 않아 JSON 저장을 건너뜁니다.")
