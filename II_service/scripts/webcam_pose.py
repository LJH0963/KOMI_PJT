import cv2
import numpy as np
import torch
import threading
from ultralytics import YOLO
from datetime import datetime

# 🔹 스레드 안전성을 위한 Lock 추가
pose_data_lock = threading.Lock()
cv2.ocl.setUseOpenCL(False)  # OpenCL 사용 안 함 (불필요한 로그 방지)

# 🔹 최신 감지된 포즈 데이터를 저장하는 전역 변수
latest_pose_data = {"status": "waiting", "pose": [], "timestamp": None}

# 📌 YOLO 모델 로드
model = YOLO("./yolov8n-pose.pt")

# 📌 웹캠 프로세스 실행 함수 (스레드에서 실행)
def capture_webcam():
    global latest_pose_data
    vcap = cv2.VideoCapture(0)

    if not vcap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        results = model(frame)

        # 감지된 포즈 데이터 저장
        pose_data = []
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()
            scores = result.keypoints.conf.cpu().numpy()

            keypoints_list = [
                {"id": i, "x": int(kp[0]), "y": int(kp[1]), "confidence": float(score)}
                for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])) if score > 0.5
            ]
            pose_data.append({"person_id": 1, "keypoints": keypoints_list})

        # 🔹 최신 포즈 데이터 갱신
        with pose_data_lock:
            latest_pose_data = {
                "status": "success",
                "pose": pose_data,
                "timestamp": datetime.utcnow().isoformat()
            }

        # 감지된 결과 화면 출력
        cv2.imshow("YOLO Pose Estimation", frame)

        # ESC 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    vcap.release()
    cv2.destroyAllWindows()

# 🔹 FastAPI에서 import 시 자동 실행되지 않도록 변경
def start_webcam_thread():
    thread = threading.Thread(target=capture_webcam, daemon=True)
    thread.start()

# python ./II_service/scripts/webcam_pose.py