import os
import pandas as pd
from utils import *

def evaluate_pose_side(answer_dir, target_dir, output_csv_path):
    answer_files = sorted([f for f in os.listdir(answer_dir) if f.endswith('.json')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.json')])
    matched = [(os.path.join(answer_dir, f), os.path.join(target_dir, f)) for f in answer_files if f in target_files]

    records = []

    for ans_path, tgt_path in matched:
        file_name = os.path.basename(ans_path)
        kps1 = load_keypoints_from_json(ans_path)  # 기준
        kps2 = load_keypoints_from_json(tgt_path)  # 평가 대상

        hip_angle_diff = angle_difference(kps1, kps2, 'left_shoulder', 'left_hip', 'left_knee') or 999
        knee_angle_diff = angle_difference(kps1, kps2, 'left_hip', 'left_knee', 'left_ankle') or 999

        slope_ref = torso_slope(kps1)
        slope_eval = torso_slope(kps2)
        slope_diff = abs(slope_ref - slope_eval) if slope_ref is not None and slope_eval is not None else 999

        dist_ref = knee_ankle_distance(kps1)
        dist_eval = knee_ankle_distance(kps2)
        knee_forward_ok = compare_relative(dist_ref, dist_eval)

        pass_hip_angle = hip_angle_diff <= 15
        pass_knee_angle = knee_angle_diff <= 15
        pass_slope = slope_diff <= 10
        pass_knee_position = knee_forward_ok

        result = "pass" if all([pass_hip_angle, pass_knee_angle, pass_slope, pass_knee_position]) else "fail"

        records.append({
            "file_name": file_name,
            "hip_angle_diff": round(hip_angle_diff, 2),
            "knee_angle_diff": round(knee_angle_diff, 2),
            "torso_slope_ref": round(slope_ref, 2) if slope_ref else None,
            "torso_slope_eval": round(slope_eval, 2) if slope_eval else None,
            "torso_slope_diff": round(slope_diff, 2),
            "knee_ankle_dist_ref": round(dist_ref, 2) if dist_ref else None,
            "knee_ankle_dist_eval": round(dist_eval, 2) if dist_eval else None,
            "knee_position_pass": pass_knee_position,
            "result": result
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ 측면 평가 완료! 결과 저장 경로: {output_csv_path}")

if __name__ == "__main__":
    answer_dir = "C:/Users/user/Desktop/img_output/squat/front_json/for_compare"
    target_dir = "C:/Users/user/Desktop/img_output/squat/web/new/front_json"
    output_csv_path = "C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/output_csv"
    evaluate_pose_side(answer_dir, target_dir, output_csv_path)