import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO

# TKinter의 GUI 숨기기
root = tk.TK()
root.withdraw()

# 이미지 폴더 먼저 선택
image_folder = filedialog.askdirectory(title="이미지 폴더를 선택하세요")

# 이미지가 없을 경우에 대한 설정
if not image_folder:
    print("폴더 내 이미지 파일이 없습니다. 프로그램을 종료합니다.")
    exit()

# 결과 저장 폴더 선택
output_folder = filedialog.askdirectory(title='저장될 폴더를 선택하세요')

# 출력 폴더를 설정하지 않을 경우?
if not output_folder:
    print("결과를 저장할 폴더를 선택하지 않았습니다. 프로그램을 종료할겁니다.")
    exit()

# YOLO-Pose 모델 로드
yolo_model = YOLO("yolo11x-pose.pt")

# cvs 파일 저장을 위한 빈 리스트 생성
csv_data = []

# 이미지 확장자 지정
image_extensions = [".jpg", ".jpeg", ".png"]

# 폴더 내의 모든 이미지 파일 찾기
image_paths = []
for root_dir, _, files in os.walk(image_folder):       # 현재 경로 / 하위 폴더 리스트(사용 X)/ 폴더 내 모든 파일
    for file in files:
        if file.lower().endswith(image_extensions):
            image_paths.append(os.path.join(root_dir, file))

# 처리할 이미지가 없는 경우 = 경로 내에서 수행이 안될 경우
if not image_paths:
    print("선택한 폴더 내에 처리할 이미지가 없습니다.")
    exit()

# 이미지 처리
for image_path in image_paths:
    # 이미지 파일명 추출
    image_name = os.path.basename(image_path)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"{image_name}을 불러올 수 없습니다.")
        continue

# YOLO-Pose 모델을 사용하여 포즈 감지
results = yolo_model(image)

# 관절(Keypoints) 좌표 추출 및 시각화
for result in results:
    keypoints = result.keypoints.xy.cpu().numpy()  # 좌표 변환
    scores = result.keypoints.conf.cpu().numpy()  # 신뢰도 변환

    for kp, score in zip(keypoints[0], scores[0]):
            x, y = int(kp[0]), int(kp[1])
            conf = float(score)

            csv_data.append([image, x, y, conf])

            if conf > 0.5:
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 빨간색 점으로 표시

    # 결과 이미지 저장
    output_image_path = os.path.join(output_folder, f"result_{image_name}")
    cv2.imwrite(output_image_path, image)
    print(f"결과 이미지 저장 완료: {output_image_path}")

    # csv 저장
    csv_file_path = os.path.join(output_folder, "keypoints_result.csv")

# 기존 파일이 있는 경우 추가 저장
if os.path.exists(csv_file_path):
    existing_df = pd.read_csv(csv_file_path)
    new_df = pd.DataFrame(csv_data, columns=["image_name", "x", "y", "confidence"])
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    final_df = pd.DataFrame(csv_data, columns=["image_name", "x", "y", "confidence"])

final_df.to_csv(csv_file_path, index=False)
print(f"CSV 파일 저장 완료: {csv_file_path}")