from utils import PoseEstimator

# 모델 불러오기
model = PoseEstimator("./tests/KHS/models/yolov8n-pose.pt")

# 실시간 Pose 데이터 수집하기
model.capture_image_detecting()