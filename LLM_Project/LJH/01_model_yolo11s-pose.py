import cv2
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt

image_path = "LJH/data/sample.jpg"

from ultralytics import YOLO

# YOLO-Pose 모델 로드
yolo_model = YOLO("yolo11x-pose.pt")

# 이미지 로드
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Cannot load image from {image_path}")
    exit()

start = datetime.now()  ### 속도 확인 ###

# YOLO-Pose 모델을 사용하여 포즈 감지
results = yolo_model(image)

# 관절(Keypoints) 좌표 추출 및 시각화
for result in results:
    keypoints = result.keypoints.xy.cpu().numpy()  # 좌표 변환
    scores = result.keypoints.conf.cpu().numpy()  # 신뢰도 변환

    for kp, score in zip(keypoints[0], scores[0]):
        if score > 0.5:  # 신뢰도 50% 이상
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 녹색 점으로 표시

# 시각화된 이미지 출력
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

print("소요시간:", datetime.now() - start)  ### 속도 확인 ###
