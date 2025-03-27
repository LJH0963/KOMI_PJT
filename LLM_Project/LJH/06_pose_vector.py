import os
import json
import numpy as np
import pandas as pd
from tkinter import filedialog, Tk

# =======================
# 1. 유틸 함수 정의
# =======================

def extract_knees_vector_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keypoints = data.get("keypoints", [])

    if len(keypoints) < 15:
        return None

    # 13 = left_knee, 14 = right_knee
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    vec = []
    for knee in [left_knee, right_knee]:
        x = knee['x'] if knee['x'] is not None else 0
        y = knee['y'] if knee['y'] is not None else 0
        vec.extend([x, y])
    return np.array(vec)


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

def evaluate_pose_similarity(
        answer_dir,
        target_dir,
        threshold=0.7,
        output_path='output_csv/pose_similarity_result.csv'
        ):
    matched_files = get_matched_json_files(answer_dir, target_dir)
    results = []

    for ans_path, tgt_path in matched_files:
        vec1 = extract_knees_vector_from_json(ans_path)
        vec2 = extract_knees_vector_from_json(tgt_path)

        if vec1 is None or vec2 is None:
            continue

        similarity = cosine_similarity(vec1, vec2)
        if similarity < threshold:
            results.append({
                'file_name': os.path.basename(ans_path),
                'similarity': round(similarity, 4)
            })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"✅ 유사도 기준 미달 프레임 {len(df)}개 저장 완료: {output_path}")


# =======================
# 4. 실행 로직
# =======================

if __name__ == '__main__':
    Tk().withdraw()
    answer_dir = filedialog.askdirectory(title="정답 JSON 폴더 선택")
    target_dir = filedialog.askdirectory(title="영상 JSON 폴더 선택")

    if not answer_dir or not target_dir:
        print("❌ 경로 선택이 취소되었습니다.")
    else:
        evaluate_pose_similarity(answer_dir, target_dir, threshold=0.7)
