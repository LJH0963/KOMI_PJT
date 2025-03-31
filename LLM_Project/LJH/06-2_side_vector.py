import os
import json
from utils import *

# JSON 형태로 평가 결과 저장 (컬럼별 key로 분리)
def evaluate_pose_front_by_angles_json(answer_dir, target_dir, output_json_path):
    answer_files = sorted([f for f in os.listdir(answer_dir) if f.endswith('.json')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.json')])
    matched = [(os.path.join(answer_dir, f), os.path.join(target_dir, f)) for f in answer_files if f in target_files]

    result_dict = {}

    for ans_path, tgt_path in matched:
        file_name = os.path.basename(ans_path)
        kps1 = load_keypoints_from_json(ans_path)  # 정답
        kps2 = load_keypoints_from_json(tgt_path)  # 평가 대상

        # 고관절 각도
        left_hip_ref = compute_angle(get_point(kps1, 'left_shoulder'), get_point(kps1, 'left_hip'), get_point(kps1, 'left_knee'))
        right_hip_ref = compute_angle(get_point(kps1, 'right_shoulder'), get_point(kps1, 'right_hip'), get_point(kps1, 'right_knee'))
        left_hip_eval = compute_angle(get_point(kps2, 'left_shoulder'), get_point(kps2, 'left_hip'), get_point(kps2, 'left_knee'))
        right_hip_eval = compute_angle(get_point(kps2, 'right_shoulder'), get_point(kps2, 'right_hip'), get_point(kps2, 'right_knee'))

        # 무릎 각도
        left_knee_ref = compute_angle(get_point(kps1, 'left_hip'), get_point(kps1, 'left_knee'), get_point(kps1, 'left_ankle'))
        right_knee_ref = compute_angle(get_point(kps1, 'right_hip'), get_point(kps1, 'right_knee'), get_point(kps1, 'right_ankle'))
        left_knee_eval = compute_angle(get_point(kps2, 'left_hip'), get_point(kps2, 'left_knee'), get_point(kps2, 'left_ankle'))
        right_knee_eval = compute_angle(get_point(kps2, 'right_hip'), get_point(kps2, 'right_knee'), get_point(kps2, 'right_ankle'))

        # 각도 차이 계산
        left_hip_diff = abs(left_hip_ref - left_hip_eval) if left_hip_ref and left_hip_eval else 999
        right_hip_diff = abs(right_hip_ref - right_hip_eval) if right_hip_ref and right_hip_eval else 999
        left_knee_diff = abs(left_knee_ref - left_knee_eval) if left_knee_ref and left_knee_eval else 999
        right_knee_diff = abs(right_knee_ref - right_knee_eval) if right_knee_ref and right_knee_eval else 999

        # 통과 여부
        pass_left_hip = 1 if left_hip_diff <= 15 else 0
        pass_right_hip = 1 if right_hip_diff <= 15 else 0
        pass_left_knee = 1 if left_knee_diff <= 15 else 0
        pass_right_knee = 1 if right_knee_diff <= 15 else 0

        # 유사도 계산
        ref_vec = [left_hip_ref, right_hip_ref, left_knee_ref, right_knee_ref]
        eval_vec = [left_hip_eval, right_hip_eval, left_knee_eval, right_knee_eval]

        if None in ref_vec or None in eval_vec:
            sim_score = 0.0
        else:
            sim_score = cosine_similarity(ref_vec, eval_vec)

        # 실패한 부위 기록
        failed_parts = []
        if not pass_left_hip:
            failed_parts.append("left_hip")
        if not pass_right_hip:
            failed_parts.append("right_hip")
        if not pass_left_knee:
            failed_parts.append("left_knee")
        if not pass_right_knee:
            failed_parts.append("right_knee")

        # 결과 딕셔너리 저장
        result_dict[file_name] = {
            "left_hip_angle_diff": round(left_hip_diff, 2),
            "right_hip_angle_diff": round(right_hip_diff, 2),
            "left_knee_angle_diff": round(left_knee_diff, 2),
            "right_knee_angle_diff": round(right_knee_diff, 2),
            "pass_left_hip": pass_left_hip,
            "pass_right_hip": pass_right_hip,
            "pass_left_knee": pass_left_knee,
            "pass_right_knee": pass_right_knee,
            "cosine_similarity": round(sim_score, 4),
            "failed_parts" : failed_parts
        }

    # JSON 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"✅ 각도 기반 정면 평가 결과 JSON 저장 완료! 경로: {output_json_path}")

if __name__ == "__main__":
    answer_dir = "C:/Users/user/Desktop/img_output/squat/side_json/for_compare"
    target_dir = "C:/Users/user/Desktop/img_output/squat/web/new/side_json"
    output_json_path = "C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/output_json/sidepose_angle_eval.json"
    evaluate_pose_front_by_angles_json(answer_dir, target_dir, output_json_path)
