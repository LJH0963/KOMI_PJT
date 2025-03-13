import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from tqdm import tqdm  # 진행 상황 표시용

# 📌 1. YOLO-Pose 모델 로드
model = YOLO("yolov8n-pose.pt")  # YOLO-Pose 경량 모델

# 📌 2. 이미지가 저장된 디렉토리 경로 설정
image_dir = "data/jeonsomi/"
output_csv = "data/jeonsomi_pose_data.csv"

# 📌 3. 저장할 데이터를 담을 리스트
pose_data_list = []

# 📌 4. 디렉토리 내 이미지 파일 리스트 가져오기
image_files = [f"jeonsomi{i}.jpg" for i in range(1, 388)]  # jeonsomi1.jpg ~ jeonsomi387.jpg

# 📌 5. 이미지별로 포즈 감지 수행
for image_name in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))

    if image is None:
        print(f"❌ Error: Cannot load image {image_name}")
        continue

    # YOLO-Pose 모델 실행
    results = model(image)

    for person_id, result in enumerate(results):
        keypoints = result.keypoints.xy.cpu().numpy()  # 🟢 GPU → CPU 변환
        scores = result.keypoints.conf.cpu().numpy()  # 🟢 신뢰도 값도 CPU 변환

        # 📌 6. 좌표 데이터 정리
        for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])):  
            if score > 0.5:  # 신뢰도 50% 이상인 경우만 저장
                pose_data_list.append([
                    image_name,  # 이미지 파일명
                    person_id + 1,  # 감지된 사람 ID
                    i,  # 관절 ID
                    int(kp[0]),  # x 좌표
                    int(kp[1]),  # y 좌표
                    float(score)  # 신뢰도
                ])

# 📌 7. 데이터프레임 생성 및 CSV 저장
columns = ["image_name", "person_id", "keypoint_id", "x", "y", "confidence"]
pose_df = pd.DataFrame(pose_data_list, columns=columns)

pose_df.to_csv(output_csv, index=False)
print(f"✅ CSV 저장 완료: {output_csv}")
