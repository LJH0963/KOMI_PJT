import os
import pandas as pd
from utils import *

# ✅ 정면 평가 전용 함수들
def get_symmetry_gap(kps, part):
    l = get_point(kps, f'left_{part}')
    r = get_point(kps, f'right_{part}')
    if l and r:
        return abs(l[0] - r[0])
    return None

def knee_gap(kps):
    l = get_point(kps, 'left_knee')
    r = get_point(kps, 'right_knee')
    if l and r:
        return abs(l[0] - r[0])
    return None

def shoulder_height_diff(kps):
    l = get_point(kps, 'left_shoulder')
    r = get_point(kps, 'right_shoulder')
    if l and r:
        return abs(l[1] - r[1])
    return None

def evaluate_pose_front(answer_dir, target_dir, output_csv_path):
    answer_files = sorted([f for f in os.listdir(answer_dir) if f.endswith('.json')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.json')])
    matched = [(os.path.join(answer_dir, f), os.path.join(target_dir, f)) for f in answer_files if f in target_files]

    records = []

    for ans_path, tgt_path in matched:
        file_name = os.path.basename(ans_path)
        kps1 = load_keypoints_from_json(ans_path)
        kps2 = load_keypoints_from_json(tgt_path)

        vec1 = extract_vector(kps1)
        vec2 = extract_vector(kps2)
        cos_sim = cosine_similarity(vec1, vec2) if vec1 is not None and vec2 is not None else 0.0

        knee_angle_diff = angle_difference(kps1, kps2, 'hip', 'knee', 'ankle') or 999
        hip_angle_diff = angle_difference(kps1, kps2, 'shoulder', 'hip', 'knee') or 999

        sym_shoulder_ref = get_symmetry_gap(kps1, 'shoulder')
        sym_knee_ref = get_symmetry_gap(kps1, 'knee')
        sym_ankle_ref = get_symmetry_gap(kps1, 'ankle')
        sym_shoulder = get_symmetry_gap(kps2, 'shoulder')
        sym_knee = get_symmetry_gap(kps2, 'knee')
        sym_ankle = get_symmetry_gap(kps2, 'ankle')

        knee_gap_ref = knee_gap(kps1)
        knee_gap_eval = knee_gap(kps2)
        shoulder_y_diff = shoulder_height_diff(kps2) or 999

        pass_cos = cos_sim >= 0.98
        pass_knee_angle = knee_angle_diff <= 15
        pass_hip_angle = hip_angle_diff <= 15
        pass_sym_shoulder = compare_relative(sym_shoulder_ref, sym_shoulder)
        pass_sym_knee = compare_relative(sym_knee_ref, sym_knee)
        pass_sym_ankle = compare_relative(sym_ankle_ref, sym_ankle)
        pass_knee_gap = compare_relative(knee_gap_ref, knee_gap_eval)
        pass_shoulder_y = shoulder_y_diff <= 20

        pass_list = [pass_cos, pass_knee_angle, pass_hip_angle, pass_sym_shoulder,
                     pass_sym_knee, pass_sym_ankle, pass_knee_gap, pass_shoulder_y]
        result = "pass" if all(pass_list) else "fail"

        records.append({
            "file_name": file_name,
            "cosine_similarity": round(cos_sim, 4),
            "knee_angle_diff": round(knee_angle_diff, 2),
            "hip_angle_diff": round(hip_angle_diff, 2),
            "shoulder_symmetry_diff": round(relative_diff(sym_shoulder_ref, sym_shoulder)*100, 2) if sym_shoulder_ref else None,
            "knee_symmetry_diff": round(relative_diff(sym_knee_ref, sym_knee)*100, 2) if sym_knee_ref else None,
            "ankle_symmetry_diff": round(relative_diff(sym_ankle_ref, sym_ankle)*100, 2) if sym_ankle_ref else None,
            "knee_gap_diff": round(relative_diff(knee_gap_ref, knee_gap_eval)*100, 2) if knee_gap_ref else None,
            "shoulder_height_diff_y": round(shoulder_y_diff, 2),
            "result": result
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ 정면 평가 완료! 결과 저장 경로: {output_csv_path}")

if __name__ == "__main__":
    answer_dir = "C:/Users/user/Desktop/img_output/squat/front_json/for_compare"
    target_dir = "C:/Users/user/Desktop/img_output/squat/web/new/front_json"
    output_csv_path = "C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/output_csv/front_pose_result.csv"
    evaluate_pose_front(answer_dir, target_dir, output_csv_path)