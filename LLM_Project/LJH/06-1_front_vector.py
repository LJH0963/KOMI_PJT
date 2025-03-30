import os
import pandas as pd
from utils import *

# 정면 평가 전용 - 각도 기반 평가 (고관절, 무릎) + 유사도

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
        left_hip_ref = compute_angle(get_point(kps1, 'left_shoulder'), get_point(kps1, 'left_hip'), get_point(kps1, 'left_knee'))
        right_hip_ref = compute_angle(get_point(kps1, 'right_shoulder'), get_point(kps1, 'right_hip'), get_point(kps1, 'right_knee'))
        left_hip_eval = compute_angle(get_point(kps2, 'left_shoulder'), get_point(kps2, 'left_hip'), get_point(kps2, 'left_knee'))
        right_hip_eval = compute_angle(get_point(kps2, 'right_shoulder'), get_point(kps2, 'right_hip'), get_point(kps2, 'right_knee'))

        # 무릎 각도 (hip-knee-ankle)
        left_knee_ref = compute_angle(get_point(kps1, 'left_hip'), get_point(kps1, 'left_knee'), get_point(kps1, 'left_ankle'))
        right_knee_ref = compute_angle(get_point(kps1, 'right_hip'), get_point(kps1, 'right_knee'), get_point(kps1, 'right_ankle'))
        left_knee_eval = compute_angle(get_point(kps2, 'left_hip'), get_point(kps2, 'left_knee'), get_point(kps2, 'left_ankle'))
        right_knee_eval = compute_angle(get_point(kps2, 'right_hip'), get_point(kps2, 'right_knee'), get_point(kps2, 'right_ankle'))

        # 각도 차이 및 방향성 / 양수 = 더 많이 굽힘, 음수 = 덜 굽힘
        left_hip_delta = left_hip_eval - left_hip_ref if left_hip_ref and left_hip_eval else None
        right_hip_delta = right_hip_eval - right_hip_ref if right_hip_ref and right_hip_eval else None
        left_knee_delta = left_knee_eval - left_knee_ref if left_knee_ref and left_knee_eval else None
        right_knee_delta = right_knee_eval - right_knee_ref if right_knee_ref and right_knee_eval else None

        left_hip_diff = abs(left_hip_delta) if left_hip_delta is not None else 999
        right_hip_diff = abs(right_hip_delta) if right_hip_delta is not None else 999
        left_knee_diff = abs(left_knee_delta) if left_knee_delta is not None else 999
        right_knee_diff = abs(right_knee_delta) if right_knee_delta is not None else 999

        # pass 기준: 15도 이하 차이
        pass_left_hip = 1 if left_hip_diff <= 15 else 0
        pass_right_hip = 1 if right_hip_diff <= 15 else 0
        pass_left_knee = 1 if left_knee_diff <= 15 else 0
        pass_right_knee = 1 if right_knee_diff <= 15 else 0

        # angle vector 유사도
        ref_vec = [left_hip_ref, right_hip_ref, left_knee_ref, right_knee_ref]
        eval_vec = [left_hip_eval, right_hip_eval, left_knee_eval, right_knee_eval]

        if None in ref_vec or None in eval_vec:
            sim_score = 0.0
        else:
            sim_score = cosine_similarity(ref_vec, eval_vec)

        records.append({
            "file_name": file_name,
            "left_hip_ref": round(left_hip_ref, 2) if left_hip_ref else None,
            "left_hip_eval": round(left_hip_eval, 2) if left_hip_eval else None,
            "left_hip_delta": round(left_hip_delta, 2) if left_hip_delta is not None else None,
            "left_hip_angle_diff": round(left_hip_diff, 2),

            "right_hip_ref": round(right_hip_ref, 2) if right_hip_ref else None,
            "right_hip_eval": round(right_hip_eval, 2) if right_hip_eval else None,
            "right_hip_delta": round(right_hip_delta, 2) if right_hip_delta is not None else None,
            "right_hip_angle_diff": round(right_hip_diff, 2),

            "left_knee_ref": round(left_knee_ref, 2) if left_knee_ref else None,
            "left_knee_eval": round(left_knee_eval, 2) if left_knee_eval else None,
            "left_knee_delta": round(left_knee_delta, 2) if left_knee_delta is not None else None,
            "left_knee_angle_diff": round(left_knee_diff, 2),

            "right_knee_ref": round(right_knee_ref, 2) if right_knee_ref else None,
            "right_knee_eval": round(right_knee_eval, 2) if right_knee_eval else None,
            "right_knee_delta": round(right_knee_delta, 2) if right_knee_delta is not None else None,
            "right_knee_angle_diff": round(right_knee_diff, 2),

            "pass_left_hip": pass_left_hip,
            "pass_right_hip": pass_right_hip,
            "pass_left_knee": pass_left_knee,
            "pass_right_knee": pass_right_knee,
            "cosine_similarity": round(sim_score, 4)
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ 각도 기반 정면 평가 + 유사도 완료! 결과 저장 경로: {output_csv_path}")

if __name__ == "__main__":
    answer_dir = "C:/Users/user/Desktop/img_output/squat/front_json/for_compare"
    target_dir = "C:/Users/user/Desktop/img_output/squat/web/new/front_json"
    output_csv_path = "C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/output_csv/front_pose_angle_eval.csv"
    evaluate_pose_front_by_angles(answer_dir, target_dir, output_csv_path)
