import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation

# 📌 1. GPU 설정 (사용 가능하면 CUDA, 아니면 CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 📌 2. ViTPose 모델 및 프로세서 로드
model_name = "usyd-community/vitpose-base-simple"
pose_processor = AutoProcessor.from_pretrained(model_name)
pose_model = VitPoseForPoseEstimation.from_pretrained(model_name).to(device)
pose_model.eval()

# 📌 3. 동영상 로드
video_path = "./tests/KSG/data/sample1_360.mp4"  # 입력 동영상 파일
output_path = "./tests/KSG/data/sample1_360_out.mp4"  # 결과 저장 파일

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# COCO 데이터셋 기준의 관절 연결 정보
skeleton = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # 팔 (오른쪽, 왼쪽)
    (11, 13), (13, 15), (12, 14), (14, 16),  # 다리 (오른쪽, 왼쪽)
    (5, 6), (11, 12), (5, 11), (6, 12)  # 몸통 연결
]

# 📌 4. 동영상 프레임별 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 📌 4-1. OpenCV 프레임을 PIL 이미지로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)

    # 📌 4-2. ViTPose 모델 입력 전처리 (boxes 수정)
    boxes = [[0, 0, frame_width, frame_height]]  # (x1, y1, x2, y2) 형식
    inputs_pose = pose_processor(images=image_pil, boxes=boxes, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_pose = pose_model(**inputs_pose)

    # 포즈 데이터 후처리
    pose_results = pose_processor.post_process_pose_estimation(outputs_pose, boxes=boxes, threshold=0.3)
    image_pose_result = pose_results[0]  # 첫 번째 이미지의 결과

    # 📌 4-3. 포즈 시각화
    for person_pose in image_pose_result:
        for keypoint, label, score in zip(person_pose["keypoints"], person_pose["labels"], person_pose["scores"]):
            if score.item() > 0.5:  # 신뢰도 50% 이상인 경우만 시각화
                x, y = int(keypoint[0].item()), int(keypoint[1].item())
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # 관절 점 시각화

    # 📌 4-4. 관절 연결선 (스켈레톤) 시각화
    for pt1, pt2 in skeleton:
        if len(image_pose_result) > 0:
            keypoints = image_pose_result[0]["keypoints"]
            if keypoints[pt1][2] > 0.5 and keypoints[pt2][2] > 0.5:
                x1, y1 = int(keypoints[pt1][0]), int(keypoints[pt1][1])
                x2, y2 = int(keypoints[pt2][0]), int(keypoints[pt2][1])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 관절 연결선 시각화

    # 📌 4-5. 결과 프레임 저장
    out.write(frame)

    # 📌 4-6. 화면 출력 (실시간 보기, 'q' 키로 종료 가능)
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 📌 5. 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()
