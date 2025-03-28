import json
import numpy as np

### Trouble : 현재 백터 유사도 = 방향에 대한 판단
### --> 따라서 제대로 된 판단을 내리는 것이 어려움
### 복합적인 점수 지표의 활용이 필요할 것으로 보임
### 공통 평가 기준과 정면/측면에 해당하는 각각의 기능을 구현하는 것으로 마무리
### 해당 파일은 현재 그 기능들 중 공통되게 사용하는 기능만을 모은 utils임.

def load_keypoints_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("keypoints", [])

def get_point(kps, part_name):
    for kp in kps:
        if kp['part'] == part_name and kp['x'] is not None and kp['y'] is not None:
            return (kp['x'], kp['y'])
    return None

def extract_vector(kps):
    vec = []
    for kp in kps:
        x = kp['x'] if kp['x'] is not None else 0
        y = kp['y'] if kp['y'] is not None else 0
        vec.extend([x, y])
    return np.array(vec) if len(vec) == 34 else None

def cosine_similarity(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def compute_angle(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]])
    bc = np.array([c[0]-b[0], c[1]-b[1]])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def angle_difference(kps1, kps2, a, b, c):
    p1 = [get_point(kps1, part) for part in [a, b, c]]
    p2 = [get_point(kps2, part) for part in [a, b, c]]
    if None in p1 or None in p2:
        return None
    angle1 = compute_angle(*p1)
    angle2 = compute_angle(*p2)
    return abs(angle1 - angle2)

def torso_slope(kps):
    shoulder = get_point(kps, 'left_shoulder')
    hip = get_point(kps, 'left_hip')
    if shoulder and hip:
        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]
        return np.degrees(np.arctan2(dy, dx))
    return None

def knee_ankle_distance(kps):
    knee = get_point(kps, 'left_knee')
    ankle = get_point(kps, 'left_ankle')
    if knee and ankle:
        return abs(knee[0] - ankle[0])
    return None

def relative_diff(val1, val2):
    return abs(val1 - val2) / val1 if val1 != 0 else None

def compare_relative(val_ref, val_eval, tolerance=0.2):
    diff = relative_diff(val_ref, val_eval)
    return diff is not None and diff <= tolerance