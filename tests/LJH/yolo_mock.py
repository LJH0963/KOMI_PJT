import json
import cv2
import sys
import os
from datetime import datetime, time
from ultralytics import YOLO
import time as tm

# 모델 불러오기
model = YOLO("./tests/KHS/models/yolov8n-pose.pt")

# 웹캠 열기
vcap = cv2.VideoCapture(0)  # 기본 웹캠

if not vcap.isOpened():
    print("웹캠을 열 수 없습니다.")
    sys.exit()

# JSON 파일 및 이미지 저장 폴더 설정
json_file = "pose_data.json"
data_list = []  # JSON 데이터 리스트
img_output_dir = "tests/LJH/img_output"
os.makedirs(img_output_dir, exist_ok=True)

# 5초 대기 후 시작
print("5초 대기 중...")
tm.sleep(5)

start_time = tm.time()
frame_count = 1

while vcap.isOpened():
    ret, frame = vcap.read()
    if not ret:
        print("웹캠 프레임을 가져올 수 없습니다.")
        break
    
    # 좌우 반전
    frame = cv2.flip(frame, 1)
    
    # 원본 이미지 복사 (좌표가 없는 원본 저장용)
    original_frame = frame.copy()
    
    # 모델 실행
    results = model(frame)
    
    pose_data = []
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        scores = result.keypoints.conf.cpu().numpy()

        keypoints_list = []
        for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])):
            if score > 0.5:  # 신뢰도 50% 이상만 포함
                keypoints_list.append({
                    "id": i,
                    "x": int(kp[0]),
                    "y": int(kp[1]),
                    "confidence": float(score)
                })
                # 감지된 좌표를 화면에 표시
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

        pose_data.append({
            "person_id": 1,
            "keypoints": keypoints_list
        })
    
    # 이미지 파일명 생성
    img_filename = f"frame_{frame_count:05d}.jpg"
    img_filepath = os.path.join(img_output_dir, img_filename)
    
    # JSON 데이터 추가
    pose_response = {
        "image_name": img_filename,
        "timestamp": datetime.utcnow().isoformat(),
        "pose": pose_data
    }
    data_list.append(pose_response)
    
    # JSON 파일 저장
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    # 원본 이미지 저장 (좌표가 없는 상태)
    cv2.imwrite(img_filepath, original_frame)
    frame_count += 1
    
    # 감지된 결과 화면 출력 (좌표가 표시된 상태)
    cv2.imshow("YOLO Pose Estimation", frame)

    # 30초 경과 후 종료
    if tm.time() - start_time > 30:
        print("30초 경과, 저장 종료.")
        break

    if cv2.waitKey(1) == 27:  # ESC 키
        break

# 웹캠 종료 및 창 닫기
vcap.release()
cv2.destroyAllWindows()
