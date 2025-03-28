import os
import pandas as pd
from utils import *

# 정면 평가 전용 - 각도 기반 평가 (고관절, 무릎)
def evaluate_pose_front_by_angles(answer_dir, target_dir, output_csv_path):
    answer_files = sorted([f for f in os.listdir(answer_dir) if f.endswith('.json')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.json')])
    matched = [(os.path.join(answer_dir, f), os.path.join(target_dir, f)) for f in answer_files if f in target_files]

    records = []

    for ans_path, tgt_path in matched:
        file_name = os.path.basename(ans_path)
        kps1 = load_keypoints_from_json(ans_path)  # 정답
        kps2 = load_keypoints_from_json(tgt_path)  # 평가 대상

        # 고관절 각도 (shoulder-hip-knee)
        left_hip_diff = angle_difference(kps1, kps2, 'left_shoulder', 'left_hip', 'left_knee') or 999
        right_hip_diff = angle_difference(kps1, kps2, 'right_shoulder', 'right_hip', 'right_knee') or 999

        # 무릎 각도 (hip-knee-ankle)
        left_knee_diff = angle_difference(kps1, kps2, 'left_hip', 'left_knee', 'left_ankle') or 999
        right_knee_diff = angle_difference(kps1, kps2, 'right_hip', 'right_knee', 'right_ankle') or 999

        # pass 기준: 15도 이하 차이
        pass_left_hip = 1 if left_hip_diff <= 15 else 0
        pass_right_hip = 1 if right_hip_diff <= 15 else 0
        pass_left_knee = 1 if left_knee_diff <= 15 else 0
        pass_right_knee = 1 if right_knee_diff <= 15 else 0

        records.append({
            "file_name": file_name,
            "left_hip_angle_diff": round(left_hip_diff, 2),
            "right_hip_angle_diff": round(right_hip_diff, 2),
            "left_knee_angle_diff": round(left_knee_diff, 2),
            "right_knee_angle_diff": round(right_knee_diff, 2),
            "pass_left_hip": pass_left_hip,
            "pass_right_hip": pass_right_hip,
            "pass_left_knee": pass_left_knee,
            "pass_right_knee": pass_right_knee
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ 각도 기반 정면 평가 완료! 결과 저장 경로: {output_csv_path}")

if __name__ == "__main__":
    answer_dir = "C:/Users/user/Desktop/img_output/squat/front_json/for_compare"
    target_dir = "C:/Users/user/Desktop/img_output/squat/web/new/front_json"
    output_csv_path = "C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/output_csv/front_pose_angle_eval.csv"
    evaluate_pose_front_by_angles(answer_dir, target_dir, output_csv_path)
