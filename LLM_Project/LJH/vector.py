import os
import json
from utils import load_keypoints_from_json, compute_angle, get_point, cosine_similarity

## 반드시 utils.py 필요!
class PoseAngleEstimator:
    def __init__(self, answer_dir, target_dir, output_json_path):
        self.answer_dir = answer_dir
        self.target_dir = target_dir
        self.output_json_path = output_json_path
        self.result_dict = {}

    def evaluate(self):
        answer_files = sorted([f for f in os.listdir(self.answer_dir) if f.endswith('.json')])
        target_files = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.json')])
        matched = [(os.path.join(self.answer_dir, f), os.path.join(self.target_dir, f)) for f in answer_files if f in target_files]

        for ans_path, tgt_path in matched:
            file_name = os.path.basename(ans_path)
            kps1 = load_keypoints_from_json(ans_path)
            kps2 = load_keypoints_from_json(tgt_path)
            self.result_dict[file_name] = self.compute_result(kps1, kps2)

        self.save_to_json()

    def compute_result(self, kps1, kps2):
        def safe_diff(ref, eval):
            return abs(ref - eval) if ref and eval else 999

        def pass_check(diff):
            return 1 if diff <= 15 else 0

        # 고관절
        lh_ref = compute_angle(get_point(kps1, 'left_shoulder'), get_point(kps1, 'left_hip'), get_point(kps1, 'left_knee'))
        rh_ref = compute_angle(get_point(kps1, 'right_shoulder'), get_point(kps1, 'right_hip'), get_point(kps1, 'right_knee'))
        lh_eval = compute_angle(get_point(kps2, 'left_shoulder'), get_point(kps2, 'left_hip'), get_point(kps2, 'left_knee'))
        rh_eval = compute_angle(get_point(kps2, 'right_shoulder'), get_point(kps2, 'right_hip'), get_point(kps2, 'right_knee'))

        # 무릎
        lk_ref = compute_angle(get_point(kps1, 'left_hip'), get_point(kps1, 'left_knee'), get_point(kps1, 'left_ankle'))
        rk_ref = compute_angle(get_point(kps1, 'right_hip'), get_point(kps1, 'right_knee'), get_point(kps1, 'right_ankle'))
        lk_eval = compute_angle(get_point(kps2, 'left_hip'), get_point(kps2, 'left_knee'), get_point(kps2, 'left_ankle'))
        rk_eval = compute_angle(get_point(kps2, 'right_hip'), get_point(kps2, 'right_knee'), get_point(kps2, 'right_ankle'))

        # 차이
        lh_diff = safe_diff(lh_ref, lh_eval)
        rh_diff = safe_diff(rh_ref, rh_eval)
        lk_diff = safe_diff(lk_ref, lk_eval)
        rk_diff = safe_diff(rk_ref, rk_eval)

        # 통과 여부
        pass_lh = pass_check(lh_diff)
        pass_rh = pass_check(rh_diff)
        pass_lk = pass_check(lk_diff)
        pass_rk = pass_check(rk_diff)

        # 유사도
        ref_vec = [lh_ref, rh_ref, lk_ref, rk_ref]
        eval_vec = [lh_eval, rh_eval, lk_eval, rk_eval]
        sim_score = cosine_similarity(ref_vec, eval_vec) if None not in ref_vec + eval_vec else 0.0

        # 실패 부위
        failed_parts = []
        if not pass_lh: failed_parts.append("left_hip")
        if not pass_rh: failed_parts.append("right_hip")
        if not pass_lk: failed_parts.append("left_knee")
        if not pass_rk: failed_parts.append("right_knee")

        return {
            "left_hip_angle_diff": round(lh_diff, 2),
            "right_hip_angle_diff": round(rh_diff, 2),
            "left_knee_angle_diff": round(lk_diff, 2),
            "right_knee_angle_diff": round(rk_diff, 2),
            "pass_left_hip": pass_lh,
            "pass_right_hip": pass_rh,
            "pass_left_knee": pass_lk,
            "pass_right_knee": pass_rk,
            "cosine_similarity": round(sim_score, 4),
            "failed_parts": failed_parts
        }

    def save_to_json(self):
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.result_dict, f, indent=2, ensure_ascii=False)
        print(f"각도 기반 평가 결과 JSON 저장완료 / 경로: {self.output_json_path}")


## 실행 예시
if __name__ == "__main__":
    answer_dir = "C:/Users/user/Desktop/img_output/squat/front_json/for_compare"
    target_dir = "C:/Users/user/Desktop/img_output/squat/web/new/front_json"
    output_json_path = "C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/output_json/front_pose_angle_eval.json"

    evaluator = PoseAngleEstimator(answer_dir, target_dir, output_json_path)
    evaluator.evaluate()
