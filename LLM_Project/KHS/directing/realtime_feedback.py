# realtime_feedback.py

from feedback_rules import evaluate_squat_pose

class RealTimeEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        """최저 자세 초기화"""
        self.lowest_pose = None
        self.lowest_hip_y = float('inf')

    def update(self, pose_data: dict):
        """한 프레임마다 호출되어 최저 자세 갱신"""
        hip_y = pose_data.get("hip_y", 1.0)
        if hip_y < self.lowest_hip_y:
            self.lowest_hip_y = hip_y
            self.lowest_pose = pose_data

    def evaluate(self) -> list:
        """최저 자세에 도달했을 때 피드백 평가"""
        if self.lowest_pose:
            return evaluate_squat_pose(self.lowest_pose)
        return []


# 예시 사용 코드 (테스트용)
if __name__ == "__main__":
    import time

    evaluator = RealTimeEvaluator()

    # 가상의 실시간 Pose 데이터 스트림
    stream = [
        {"frame_id": 1, "hip_y": 0.85, "left_knee_angle": 95, "right_knee_angle": 93, "back_angle": 72, "knee_to_knee_distance": 0.3, "shoulder_width": 0.5, "hip_angle": 125},
        {"frame_id": 2, "hip_y": 0.78, "left_knee_angle": 92, "right_knee_angle": 91, "back_angle": 70, "knee_to_knee_distance": 0.29, "shoulder_width": 0.5, "hip_angle": 122},
        {"frame_id": 3, "hip_y": 0.72, "left_knee_angle": 88, "right_knee_angle": 86, "back_angle": 65, "knee_to_knee_distance": 0.26, "shoulder_width": 0.5, "hip_angle": 118},
        {"frame_id": 4, "hip_y": 0.74, "left_knee_angle": 90, "right_knee_angle": 88, "back_angle": 68, "knee_to_knee_distance": 0.27, "shoulder_width": 0.5, "hip_angle": 119},
    ]

    for pose in stream:
        evaluator.update(pose)
        time.sleep(0.033)  # 30fps 시뮬레이션

    # 스쿼트 한 세트 완료 시점에서 평가 실행
    feedback = evaluator.evaluate()

    print("\n📣 실시간 피드백 결과:")
    for msg in feedback:
        print(f"👉 {msg}")

    evaluator.reset()
