import cv2
import numpy as np
from datetime import datetime
from .config import yolo_model  # YOLO 모델 로드
from typing import List, Dict, Any, Optional, Tuple
import random
import json
import time

# 더미 키포인트 데이터 (COCO 형식)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def generate_dummy_keypoints() -> List[Dict[str, Any]]:
    """
    더미 키포인트 생성 함수
    """
    keypoints = []
    for i, name in enumerate(KEYPOINT_NAMES):
        keypoints.append({
            "id": i,
            "name": name,
            "x": random.uniform(100, 500),
            "y": random.uniform(100, 400),
            "confidence": random.uniform(0.7, 0.95)
        })
    return keypoints

# 더미 가이드 포즈 데이터
GUIDE_POSES = {
    "shoulder": {
        "description": "어깨 스트레칭 자세",
        "keypoints": generate_dummy_keypoints()
    },
    "knee": {
        "description": "무릎 스트레칭 자세",
        "keypoints": generate_dummy_keypoints()
    }
}

def detect_pose(image_data: bytes) -> Dict[str, Any]:
    """
    더미 포즈 감지 함수
    """
    # 실제 포즈 감지 로직 대신 더미 데이터 생성
    time.sleep(0.1)  # 포즈 감지 처리 시간 시뮬레이션
    
    return {
        "pose": [
            {
                "keypoints": generate_dummy_keypoints(),
                "bbox": [100, 100, 400, 400],
                "confidence": random.uniform(0.8, 0.95)
            }
        ],
        "processing_time": 0.1,
        "timestamp": time.time()
    }

def compare_poses(user_pose: Dict[str, Any], guide_pose_type: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
    """
    더미 포즈 비교 함수
    """
    # 가이드 포즈가 없으면 무작위 점수 반환
    if not guide_pose_type or guide_pose_type not in GUIDE_POSES:
        return random.uniform(50, 100), {}
    
    # 사용자 키포인트와 가이드 키포인트 간의 더미 유사도 계산
    similarity_details = {}
    for name in ["shoulders", "arms", "legs", "torso"]:
        similarity_details[name] = random.uniform(50, 100)
    
    # 전체 정확도 계산 (더미 값)
    accuracy = sum(similarity_details.values()) / len(similarity_details)
    
    return accuracy, similarity_details

def get_guide_pose(exercise_type: str) -> Dict[str, Any]:
    """
    운동 유형에 따른 가이드 포즈 반환
    """
    if exercise_type in GUIDE_POSES:
        return GUIDE_POSES[exercise_type]
    
    # 기본 가이드 포즈 반환
    return {
        "description": "기본 자세",
        "keypoints": generate_dummy_keypoints()
    }

def process_pose(image: np.ndarray):
    """
    YOLO Pose 모델을 사용하여 이미지에서 관절 포인트 감지
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
