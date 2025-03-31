# 라이브러리 불러오기
from collections import Counter
import json

# JSON 파일을 읽어서 프레임 단위 데이터를 가져오는 클래스
class PoseAnalyzer():
    def __init__(self, input_path: str):
        """입력값으로 가져올 데이터 경로를 설정"""
        self.input_path = input_path
        self.data = self.load_data()

    def load_data(self):
        """입력된 경로의 JSON 파일을 열어서 전체 데이터를 로딩하는 함수"""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def get_all_frames(self):
        """로딩된 데이터를 외부에서 접근 할 수 있도록 반환하는 함수"""
        return self.data

# 각도 차이가 기준(15도) 이상인 관절을 프레임 단위로 추출하는 클래스
class PoseEvaluator:
    def __init__(self, pose_analyzer: PoseAnalyzer, threshold: float = 15.0):
        self.data = pose_analyzer.get_all_frames()
        self.threshold = threshold
    
    def evaluate(self) -> dict:
        """frame마다 *_angle_diff값이 기준 이상인 관절을 찾고 정리하는 함수"""
        failed_frames = {}
        for frame_name, frame_data in self.data.items():
            failed_joints = []

            for key, value in frame_data.items():
                if key.endswith('_angle_diff') and abs(value) >= self.threshold:
                    failed_joints.append(key)

            if failed_joints:
                failed_frames[frame_name] = failed_joints

        return failed_frames

# 프레임별로 문제가 된 관절들을 전부 모아서 관절별로 몇 번 잘못됐는지 Count
def summarize_failed_joints(failed_dict: dict) -> Counter:
    joint_counter = Counter()
    for joints in failed_dict.values():
        for joint in joints:
            joint_counter[joint] += 1
    return joint_counter

# 위에서 구한 관절별 오류 통계를 바탕으로 LLM에 보낼 자연어 프롬프트 생성
def generate_summary_prompt(input_path: str) -> str:
    analyzer = PoseAnalyzer(input_path)
    evaluator = PoseEvaluator(analyzer)
    failed_results = evaluator.evaluate()
    joint_counter = summarize_failed_joints(failed_results)

    # 위에서 구한 관절별 오류 통계를 바탕으로 LLM에 보낼 자연어 프롬프트 생성
    part_names = {
        # Front view
        "left_hip_angle_diff": "왼쪽 고관절",
        "right_hip_angle_diff": "오른쪽 고관절",
        "left_knee_angle_diff": "왼쪽 무릎",
        "right_knee_angle_diff": "오른쪽 무릎",

        # Side view
        "left_shoulder_angle_diff": "왼쪽 어깨",
        "left_hip_angle_diff": "왼쪽 고관절",
        "left_knee_angle_diff": "왼쪽 무릎",
    }

    # 관절별 잘못된 횟수를 텍스트로 나열
    front_lines = []
    side_lines = []

    for joint, count in joint_counter.items():
        name = part_names.get(joint, joint)
        line = f"- {name}: {count}회"
        if "angle" in joint and ("left" in joint or "right" in joint):
            front_lines.append(line)
        else:
            side_lines.append(line)

    # 최종 LLM용 프롬프트 생성
    prompt = "운동 영상에서 다음 부위에 자주 문제가 발생했습니다:\n"

    if front_lines:
        prompt += "\n[정면 View 기준 문제 부위]\n" + "\n".join(front_lines)
    if side_lines:
        prompt += "\n\n[측면 View 기준 문제 부위]\n" + "\n".join(side_lines)

    prompt += (
        "\n\n이러한 문제가 왜 발생할 수 있는지, 그리고 어떻게 개선하면 좋을지 "
        "운동 전문가 입장에서 설명해 주세요."
    )
    return prompt