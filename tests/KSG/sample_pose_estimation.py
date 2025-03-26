import cv2
import numpy as np
from datetime import datetime
from komi_service.tests.config import yolo_model  # YOLO 모델 로드

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
        "timestamp": datetime.now().isoformat(),
    }

def compare_poses(user_keypoints, guide_keypoints):
    """
    📌 사용자 포즈와 가이드 포즈 비교 (코사인 유사도 사용)
    - 입력: 사용자 키포인트 목록, 가이드 키포인트 목록
    - 출력: 정확도 점수 (0-100)
    - 계산 방식: 관절 벡터의 코사인 유사도
    """
    # 가이드 키포인트가 없는 경우 임의의 정확도 반환 (데모용)
    if not guide_keypoints or len(guide_keypoints) == 0:
        return np.clip(np.random.normal(75, 15), 50, 100)
    
    # 사용자 키포인트를 딕셔너리로 변환 (ID 기준)
    user_kp_dict = {kp["id"]: (kp["x"], kp["y"]) for kp in user_keypoints}
    guide_kp_dict = {kp["id"]: (kp["x"], kp["y"]) for kp in guide_keypoints}
    
    # 주요 관절 벡터 정의 (COCO 키포인트 포맷 기준)
    # 각 관절 그룹은 (시작점, 끝점) 형태로 정의
    joint_vectors = [
        # 오른팔 (어깨-팔꿈치, 팔꿈치-손목)
        [(6, 8), (8, 10)],
        # 왼팔 (어깨-팔꿈치, 팔꿈치-손목)
        [(5, 7), (7, 9)],
        # 오른다리 (엉덩이-무릎, 무릎-발목)
        [(12, 14), (14, 16)],
        # 왼다리 (엉덩이-무릎, 무릎-발목)
        [(11, 13), (13, 15)],
        # 몸통 (어깨-골반)
        [(6, 12), (5, 11)]
    ]
    
    # 벡터 유사도 계산을 위한 키포인트 쌍 검증
    valid_vectors = []
    for vector_group in joint_vectors:
        for start_id, end_id in vector_group:
            if start_id in user_kp_dict and end_id in user_kp_dict and start_id in guide_kp_dict and end_id in guide_kp_dict:
                valid_vectors.append((start_id, end_id))
    
    if not valid_vectors:
        return 50.0  # 유효한 벡터가 없으면 기본값 반환
    
    # 코사인 유사도 계산
    similarities = []
    for start_id, end_id in valid_vectors:
        # 사용자 벡터
        user_start = np.array(user_kp_dict[start_id])
        user_end = np.array(user_kp_dict[end_id])
        user_vector = user_end - user_start
        
        # 가이드 벡터
        guide_start = np.array(guide_kp_dict[start_id])
        guide_end = np.array(guide_kp_dict[end_id])
        guide_vector = guide_end - guide_start
        
        # 벡터의 크기가 0이 아닌지 확인
        if np.linalg.norm(user_vector) > 0 and np.linalg.norm(guide_vector) > 0:
            # 코사인 유사도 계산
            cos_sim = np.dot(user_vector, guide_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(guide_vector))
            # 유사도 범위는 -1에서 1이므로, 0에서 1 범위로 정규화
            normalized_sim = (cos_sim + 1) / 2
            similarities.append(normalized_sim)
    
    if not similarities:
        return 50.0  # 유사도 계산 실패 시 기본값
    
    # 최종 정확도 계산 (평균 코사인 유사도)
    avg_similarity = np.mean(similarities)
    
    # 50-100 범위로 변환
    final_accuracy = 50 + (avg_similarity * 50)
    
    return final_accuracy
