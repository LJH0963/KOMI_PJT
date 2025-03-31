# feedback_rules.py

import math

def evaluate_squat_pose(pose_data: dict) -> list:
    """
    실시간으로 전달된 관절 데이터를 기반으로 개선 피드백을 생성합니다.
    Args:
        pose_data (dict): 팀원1이 전달한 관절 각도/거리 등 실시간 연산 데이터
    Returns:
        list[str]: 개선이 필요한 부분에 대한 멘트 목록
    """
    feedback = []

    # 무릎이 충분히 굽혀지지 않은 경우
    if pose_data.get("left_knee_angle", 180) > 90 or pose_data.get("right_knee_angle", 180) > 90:
        feedback.append("조금 더 앉아주세요.")

    # 허리가 너무 숙여진 경우
    if pose_data.get("back_angle", 90) < 70:
        feedback.append("허리를 너무 숙이지 마세요.")

    # 정면(front) 기준: 무릎 간격이 어깨보다 너무 좁거나 벌어진 경우
    knee_gap = pose_data.get("knee_to_knee_distance", 0)
    shoulder_width = pose_data.get("shoulder_width", 0.5)
    if knee_gap < shoulder_width * 0.8:
        feedback.append("무릎이 너무 붙었습니다. 벌려주세요.")
    elif knee_gap > shoulder_width * 1.2:
        feedback.append("무릎이 너무 벌어졌습니다. 모아주세요.")

    # 고관절 각도 확인
    if pose_data.get("hip_angle", 180) > 120:
        feedback.append("엉덩이를 더 낮춰보세요.")

    # 측면(side) 기준: 척추 중립 확인
    # shoulder, hip, knee의 좌표가 필요 (x, y)
    try:
        sx, sy = pose_data["shoulder"]
        hx, hy = pose_data["hip"]
        kx, ky = pose_data["knee"]

        # 어깨-고관절-무릎을 잇는 각도를 계산
        def angle_between_points(a, b, c):
            ab = (a[0] - b[0], a[1] - b[1])
            cb = (c[0] - b[0], c[1] - b[1])
            dot_product = ab[0] * cb[0] + ab[1] * cb[1]
            ab_mag = math.hypot(*ab)
            cb_mag = math.hypot(*cb)
            if ab_mag == 0 or cb_mag == 0:
                return 180.0
            cos_angle = dot_product / (ab_mag * cb_mag)
            angle_rad = math.acos(max(min(cos_angle, 1.0), -1.0))
            return math.degrees(angle_rad)

        spine_angle = angle_between_points((sx, sy), (hx, hy), (kx, ky))
        if abs(spine_angle - 180) > 20:  # 180도에 가까울수록 척추 중립
            feedback.append("척추 중립을 유지해 주세요.")
    except KeyError:
        pass  # 좌표가 없을 경우 평가 생략

    return feedback