from utils import PoseEstimator
from fastapi import APIRouter
import cv2

# 라우터 설정
image_yolo_router = APIRouter(prefix='/image_yolo', tags=['Image YOLO'])

# 모델 불러오기
model = PoseEstimator("./tests/KHS/models/yolov8n-pose.pt")

# Webcam에서 Frame 캡처하기
vcap = cv2.VideoCapture(0)  # 0번 카메라 (기본 웹캠) 연결

# 웹캠에서 프레임 읽기
while vcap.isOpened():

    # 웹캠 사이즈 및 해상도 설정
    vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)                        # 3을 'cv2.CAP_PROP_FRAME_WIDTH'로 변경해도 됨
    vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)                      # 4를 'cv2.CAP_PROP_FRAME_HEIGHT'로 변경해도 됨
    vcap.set(cv2.CAP_PROP_FPS, 30)                                 # 5를 'cv2.CAP_PROP_FPS'로 변경해도 됨

    ret, frame = vcap.read()

    # 꺼지는 조건 설정
    key = cv2.waitKey(1)

    # ESC : 27 (아스키코드)
    if key == 27:
        break

    if not ret:
        # 만약 정상 작동되지 않는다면 break
        print("❌ 웹캠 프레임을 가져올 수 없습니다.")
    else:
        # 좌우 반전
        frame = cv2.flip(frame, 1)

        # 좌표 감지 함수 사용
        pose_data, save_frame = model.detect_image_pose(frame)

        # 추출 이미지 저장
        cv2.imwrite('./tests/KHS/data/정답지' + '.jpg', save_frame)

# 웹캠 닫기
vcap.release()
cv2.destroyAllWindows()