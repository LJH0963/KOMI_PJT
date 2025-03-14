from utils import PoseEstimator

# 모델 불러오기
model = PoseEstimator("./tests/KHS/models/yolo11n-pose.pt")

# 실시간 Pose 데이터 수집하기
model.video_image_extraction("mediun_video_640_480", 8)