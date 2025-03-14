from utils import PoseEstimator

# 모델 불러오기
model = PoseEstimator("./tests/KHS/models/yolov8n-pose.pt")

# 웹캠 시작하기
model.start_camera(src=0)

# 실시간 Pose 데이터 수집하기
model.real_time_video_detecting()