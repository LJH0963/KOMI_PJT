import torch
from ultralytics import YOLO

# 📌 YOLO 모델 로드
MODEL_PATH = "./yolov8n-pose.pt"

# 📌 디바이스 설정 (CUDA 사용 가능 여부 확인)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 📌 YOLO 모델 초기화
yolo_model = YOLO(MODEL_PATH).to(DEVICE)
