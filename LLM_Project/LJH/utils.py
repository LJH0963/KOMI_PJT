import json
import numpy as np

# JSON에서 keypoints 불러오기
def load_keypoints_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("keypoints", [])

# 특정 관절 좌표 반환
def get_point(kps, part_name):
    for kp in kps:
        if kp['part'] == part_name and kp['x'] is not None and kp['y'] is not None:
            return (kp['x'], kp['y'])
    return None

# 관절 3점으로 각도 계산
def compute_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# 두 keypoints 세트 간 각도 차이 계산
def angle_difference(kps1, kps2, a, b, c):
    p1 = [get_point(kps1, part) for part in [a, b, c]]
    p2 = [get_point(kps2, part) for part in [a, b, c]]
    if None in p1 or None in p2:
        return None
    angle1 = compute_angle(*p1)
    angle2 = compute_angle(*p2)
    return abs(angle1 - angle2)
