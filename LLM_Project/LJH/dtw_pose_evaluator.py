import os
import json
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- utils.py functions ---
def load_keypoints_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("keypoints", [])

def get_point(kps, part_name):
    for kp in kps:
        if kp['part'] == part_name and kp['x'] is not None and kp['y'] is not None:
            return (kp['x'], kp['y'])
    return None

def compute_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def cosine_similarity(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# --- Main Class ---
class DTWPoseEvaluator:
    COCO_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                      "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                      "left_wrist", "right_wrist", "left_hip", "right_hip",
                      "left_knee", "right_knee", "left_ankle", "right_ankle"]

    def __init__(self, ref_dir, user_dir, image_dir, output_vis_dir, output_json_dir):
        self.ref_dir = ref_dir
        self.user_dir = user_dir
        self.image_dir = image_dir
        self.output_vis_dir = output_vis_dir
        self.output_json_dir = output_json_dir
        os.makedirs(self.output_vis_dir, exist_ok=True)
        os.makedirs(self.output_json_dir, exist_ok=True)

    def flatten_keypoints(self, kps_dict_list):
        return np.array([
            [kp['x'] if kp['x'] is not None else 0,
             kp['y'] if kp['y'] is not None else 0]
            for kp in kps_dict_list
        ]).flatten()

    def draw_keypoints(self, img, keypoints_dict, color):
        for kp in keypoints_dict:
            x, y = kp['x'], kp['y']
            if x is not None and y is not None:
                cv2.circle(img, (int(x), int(y)), 4, color, -1)
                cv2.putText(img, kp['part'], (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        return img

    def compute_pose_eval(self, ref_kps, user_kps):
        def safe_diff(a, b): return abs(a - b) if a and b else 999
        def pass_check(d): return 1 if d <= 15 else 0

        lh_ref = compute_angle(get_point(ref_kps, 'left_shoulder'), get_point(ref_kps, 'left_hip'), get_point(ref_kps, 'left_knee'))
        rh_ref = compute_angle(get_point(ref_kps, 'right_shoulder'), get_point(ref_kps, 'right_hip'), get_point(ref_kps, 'right_knee'))
        lk_ref = compute_angle(get_point(ref_kps, 'left_hip'), get_point(ref_kps, 'left_knee'), get_point(ref_kps, 'left_ankle'))
        rk_ref = compute_angle(get_point(ref_kps, 'right_hip'), get_point(ref_kps, 'right_knee'), get_point(ref_kps, 'right_ankle'))

        lh_usr = compute_angle(get_point(user_kps, 'left_shoulder'), get_point(user_kps, 'left_hip'), get_point(user_kps, 'left_knee'))
        rh_usr = compute_angle(get_point(user_kps, 'right_shoulder'), get_point(user_kps, 'right_hip'), get_point(user_kps, 'right_knee'))
        lk_usr = compute_angle(get_point(user_kps, 'left_hip'), get_point(user_kps, 'left_knee'), get_point(user_kps, 'left_ankle'))
        rk_usr = compute_angle(get_point(user_kps, 'right_hip'), get_point(user_kps, 'right_knee'), get_point(user_kps, 'right_ankle'))

        lh_d, rh_d = safe_diff(lh_ref, lh_usr), safe_diff(rh_ref, rh_usr)
        lk_d, rk_d = safe_diff(lk_ref, lk_usr), safe_diff(rk_ref, rk_usr)

        return {
            "left_hip_angle_diff": round(lh_d, 2),
            "right_hip_angle_diff": round(rh_d, 2),
            "left_knee_angle_diff": round(lk_d, 2),
            "right_knee_angle_diff": round(rk_d, 2),
            "pass_left_hip": pass_check(lh_d),
            "pass_right_hip": pass_check(rh_d),
            "pass_left_knee": pass_check(lk_d),
            "pass_right_knee": pass_check(rk_d),
            "cosine_similarity": round(cosine_similarity(
                [lh_ref, rh_ref, lk_ref, rk_ref],
                [lh_usr, rh_usr, lk_usr, rk_usr]
            ), 4) if None not in [lh_ref, rh_ref, lk_ref, rk_ref, lh_usr, rh_usr, lk_usr, rk_usr] else 0.0,
            "failed_parts": [p for p, passed in zip(
                ["left_hip", "right_hip", "left_knee", "right_knee"],
                [pass_check(lh_d), pass_check(rh_d), pass_check(lk_d), pass_check(rk_d)]
            ) if not passed]
        }

    def evaluate(self):
        ref_files = sorted([f for f in os.listdir(self.ref_dir) if f.endswith('.json')])
        user_files = sorted([f for f in os.listdir(self.user_dir) if f.endswith('.json')])

        # print("정답 JSON 개수:", len(ref_files))
        # print("사용자 JSON 개수:", len(user_files))

        for f in ref_files:
            kps = load_keypoints_from_json(os.path.join(self.ref_dir, f))
            if len(kps) != 17:
                print(f"⚠ keypoint 수가 부족한 파일: {f} → {len(kps)}개")

        ref_sequence = [self.flatten_keypoints(load_keypoints_from_json(os.path.join(self.ref_dir, f))) for f in ref_files]
        user_sequence = [self.flatten_keypoints(load_keypoints_from_json(os.path.join(self.user_dir, f))) for f in user_files]

        _, path = fastdtw(ref_sequence, user_sequence, dist=euclidean)

        # print("DTW 매핑 수:", len(path))
        # print("첫 5개 매핑 예시:", path[:5])

        for ref_idx, user_idx in path:
            ref_json_path = os.path.join(self.ref_dir, ref_files[ref_idx])
            user_json_path = os.path.join(self.user_dir, user_files[user_idx])
            img_path = os.path.join(self.image_dir, user_files[user_idx].replace(".json", ".jpg"))

            if not os.path.exists(img_path):
                print(f"이미지 없음: {img_path}")
                continue

            img = cv2.imread(img_path)
            ref_kps = load_keypoints_from_json(ref_json_path)
            user_kps = load_keypoints_from_json(user_json_path)

            img = self.draw_keypoints(img, ref_kps, (255, 0, 0))
            img = self.draw_keypoints(img, user_kps, (0, 0, 255))
            vis_path = os.path.join(self.output_vis_dir, f"{user_files[user_idx].replace('.json', '.jpg')}")
            cv2.imwrite(vis_path, img)

            # 이미지 확인용
            cv2.imshow("Pose Comparison", img)
            cv2.waitKey(100)
            cv2.destroyAllWindows()

            result = self.compute_pose_eval(ref_kps, user_kps)
            json_save_path = os.path.join(self.output_json_dir, user_files[user_idx])
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nDTW 기반 자세 비교 완료!\n- 시각화 저장: {self.output_vis_dir}\n- 평가 JSON 저장: {self.output_json_dir}")


# --- 실행 예시 ---
if __name__ == "__main__":
    evaluator = DTWPoseEvaluator(
        ref_dir="C:/Users/user/Desktop/img_output/squat/front_json/for_compare",
        user_dir="C:/Users/user/Desktop/img_output/squat/web/new/front_json",
        image_dir="C:/Users/user/Desktop/img_output/squat/web/new/image",
        output_vis_dir="C:/Users/user/Desktop/img_output/squat/web/yaammii/image",
        output_json_dir="C:/Users/user/Desktop/img_output/squat/web/yaammii/json"
    )
    evaluator.evaluate()