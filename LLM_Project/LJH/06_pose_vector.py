import os
import json
import numpy as np
import pandas as pd

# =======================
# 1. 유틸 함수 정의
# =======================

def extract_all_keypoints_vector_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keypoints = data.get("keypoints", [])

    vec = []
    for kp in keypoints:
        x = kp['x'] if kp['x'] is not None else 0
        y = kp['y'] if kp['y'] is not None else 0
        vec.extend([x, y])
    return np.array(vec) if len(vec) == 34 else None

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# =======================
# 2. 경로 선택 및 파일 매칭
# =======================

def get_matched_json_files(answer_dir, target_dir):
    answer_files = sorted([f for f in os.listdir(answer_dir) if f.endswith('.json')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.json')])
    matched = [(os.path.join(answer_dir, f), os.path.join(target_dir, f)) for f in answer_files if f in target_files]
    return matched

# =======================
# 3. 유사도 비교 및 저장
# =======================

def evaluate_pose_similarity(answer_dir, target_dir, threshold=0.98, output_path='C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/output_csv/pose_similarity_result.csv'):
    matched_files = get_matched_json_files(answer_dir, target_dir)
    results = []

    answer_vectors = []  # 시퀀스 벡터 저장용 (정답)
    target_vectors = []  # 시퀀스 벡터 저장용 (비교 대상)
    similarity_scores = []  # 전체 유사도 저장용

    for ans_path, tgt_path in matched_files:
        vec1 = extract_all_keypoints_vector_from_json(ans_path)
        vec2 = extract_all_keypoints_vector_from_json(tgt_path)

        if vec1 is None or vec2 is None:
            continue

        answer_vectors.append(vec1.tolist())
        target_vectors.append(vec2.tolist())

        similarity = cosine_similarity(vec1, vec2)
        similarity_scores.append(similarity)

        if similarity < threshold:
            results.append({
                'file_name': os.path.basename(ans_path),
                'similarity': round(similarity, 4)
            })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"✅ 유사도 기준 미달 프레임 {len(df)}개 저장 완료: {output_path}")

    # 전체 유사도 저장
    similarity_df = pd.DataFrame({
        'file_name': [os.path.basename(p[0]) for p in matched_files[:len(similarity_scores)]],
        'similarity': similarity_scores
    })
    similarity_df.to_csv('C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/output_csv/all_similarity_scores.csv', index=False)
    print("📊 전체 프레임 유사도 저장 완료: all_similarity_scores.csv")

# =======================
# 4. 실행 로직
# =======================

if __name__ == '__main__':
    answer_dir = "C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/data/dummy/front_json"
    target_dir = "C:/Users/user/Desktop/new/json"

    if not answer_dir or not target_dir:
        print("❌ 경로 선택이 취소되었습니다.")
    else:
        evaluate_pose_similarity(answer_dir, target_dir, threshold=0.98)