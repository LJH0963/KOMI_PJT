import cv2
import numpy as np
from datetime import datetime
from .config import yolo_model  # YOLO 모델 로드

def process_pose(image: np.ndarray):
    """
    📌 YOLO Pose 모델을 사용하여 이미지에서 포즈 감지
    - 입력: OpenCV 이미지 (numpy.ndarray)
    - 출력: 포즈 데이터 (딕셔너리 형태)
    """
    results = yolo_model(image, verbose=False)
    pose_data = []

    for result in results:
        if result.keypoints is None or result.keypoints.xy is None or result.keypoints.conf is None:
            continue  # 포즈 감지 실패 시 스킵

        keypoints = result.keypoints.xy.cpu().numpy()
        scores = result.keypoints.conf.cpu().numpy()

        keypoints_list = [
            {"id": i, "x": int(kp[0]), "y": int(kp[1]), "confidence": float(score)}
            for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])) if score > 0.5
        ]
        pose_data.append({"person_id": 1, "keypoints": keypoints_list})

    return {
        "status": "success",
        "pose": pose_data,
        "timestamp": datetime.now(),
    }
