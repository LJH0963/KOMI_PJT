# 자세 분석 알고리즘 파일

# 이 파일은 자세 분석 알고리즘을 구현하기 위한 파일입니다. 

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math

class PoseAnalyzer:
    """
    자세 분석 알고리즘 클래스
    - 키포인트 기반 자세 분석
    - 관절 각도 계산
    - 자세 정확도 평가
    """
    
    def __init__(self):
        """
        PoseAnalyzer 초기화
        """
        # 중요 관절 부위 정의 (평가에 사용될 관절)
        self.key_joints = [
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 자세 평가에 사용될 관절 쌍
        self.joint_pairs = [
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")
        ]
    
    def calculate_joint_angles(self, keypoints: List[Dict]):
        """
        관절 각도 계산
        
        Args:
            keypoints: 키포인트 리스트
            
        Returns:
            관절 각도 딕셔너리
        """
        # 키포인트를 딕셔너리로 변환
        kp_dict = {kp["part"]: kp for kp in keypoints}
        
        angles = {}
        
        # 팔꿈치 각도 (어깨-팔꿈치-손목)
        angles["left_elbow"] = self._calculate_angle(
            kp_dict.get("left_shoulder"), 
            kp_dict.get("left_elbow"), 
            kp_dict.get("left_wrist")
        )
        
        angles["right_elbow"] = self._calculate_angle(
            kp_dict.get("right_shoulder"), 
            kp_dict.get("right_elbow"), 
            kp_dict.get("right_wrist")
        )
        
        # 무릎 각도 (엉덩이-무릎-발목)
        angles["left_knee"] = self._calculate_angle(
            kp_dict.get("left_hip"), 
            kp_dict.get("left_knee"), 
            kp_dict.get("left_ankle")
        )
        
        angles["right_knee"] = self._calculate_angle(
            kp_dict.get("right_hip"), 
            kp_dict.get("right_knee"), 
            kp_dict.get("right_ankle")
        )
        
        # 엉덩이 각도 (어깨-엉덩이-무릎)
        angles["left_hip"] = self._calculate_angle(
            kp_dict.get("left_shoulder"), 
            kp_dict.get("left_hip"), 
            kp_dict.get("left_knee")
        )
        
        angles["right_hip"] = self._calculate_angle(
            kp_dict.get("right_shoulder"), 
            kp_dict.get("right_hip"), 
            kp_dict.get("right_knee")
        )
        
        return angles
    
    def _calculate_angle(self, joint1: Dict, joint2: Dict, joint3: Dict) -> Optional[float]:
        """
        세 관절 포인트로 각도 계산
        
        Args:
            joint1, joint2, joint3: 키포인트 딕셔너리 (joint2가 각도의 중심)
            
        Returns:
            각도(도, degree) 또는 None
        """
        if not joint1 or not joint2 or not joint3:
            return None
            
        if (joint1.get("x") is None or joint1.get("y") is None or 
            joint2.get("x") is None or joint2.get("y") is None or 
            joint3.get("x") is None or joint3.get("y") is None):
            return None
            
        # 신뢰도 검사
        min_conf = 0.3
        if (joint1.get("confidence", 0) < min_conf or 
            joint2.get("confidence", 0) < min_conf or 
            joint3.get("confidence", 0) < min_conf):
            return None
        
        # 벡터 계산
        vector1 = (joint1["x"] - joint2["x"], joint1["y"] - joint2["y"])
        vector2 = (joint3["x"] - joint2["x"], joint3["y"] - joint2["y"])
        
        # 내적 계산
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        
        # 벡터 크기 계산
        mag1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        mag2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        # 0으로 나누기 방지
        if mag1 * mag2 == 0:
            return None
        
        # 코사인 값
        cos_angle = dot_product / (mag1 * mag2)
        
        # 부동소수점 오류 방지
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # 라디안에서 도로 변환
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def calculate_pose_similarity(self, reference_keypoints: List[Dict], 
                                 current_keypoints: List[Dict]) -> Dict:
        """
        참조 포즈와 현재 포즈의 유사도 계산
        
        Args:
            reference_keypoints: 참조 포즈 키포인트
            current_keypoints: 현재 포즈 키포인트
            
        Returns:
            유사도 점수 및 세부 분석 결과
        """
        # 키포인트를 딕셔너리로 변환
        ref_kp = {kp["part"]: kp for kp in reference_keypoints}
        cur_kp = {kp["part"]: kp for kp in current_keypoints}
        
        # 관절 각도 계산
        ref_angles = self.calculate_joint_angles(reference_keypoints)
        cur_angles = self.calculate_joint_angles(current_keypoints)
        
        # 각도 차이 계산
        angle_diffs = {}
        for joint, angle in ref_angles.items():
            if angle is not None and joint in cur_angles and cur_angles[joint] is not None:
                angle_diffs[joint] = abs(angle - cur_angles[joint])
        
        # 전체 유사도 점수 계산
        if not angle_diffs:
            return {
                "score": 0,
                "joint_scores": {},
                "angle_diffs": {},
                "message": "충분한 관절 정보를 찾을 수 없습니다."
            }
        
        # 각 관절에 대한 점수 계산 (각도 차이가 작을수록 높은 점수)
        joint_scores = {}
        for joint, diff in angle_diffs.items():
            # 30도 이상 차이나면 0점, 완전히 같으면 100점
            score = max(0, 100 - (diff * 100 / 30))
            joint_scores[joint] = score
        
        # 전체 점수는 각 관절 점수의 평균
        overall_score = sum(joint_scores.values()) / len(joint_scores)
        
        return {
            "score": overall_score,
            "joint_scores": joint_scores,
            "angle_diffs": angle_diffs,
            "message": self._generate_feedback(joint_scores, angle_diffs)
        }
    
    def _generate_feedback(self, joint_scores: Dict[str, float], 
                           angle_diffs: Dict[str, float]) -> str:
        """
        자세 분석 결과에 기반한 피드백 메시지 생성
        
        Args:
            joint_scores: 관절별 점수
            angle_diffs: 관절별 각도 차이
            
        Returns:
            피드백 메시지
        """
        if not joint_scores:
            return "포즈를 분석할 수 없습니다. 더 나은 포즈를 취해주세요."
        
        # 점수가 가장 낮은 관절 찾기
        worst_joints = sorted(joint_scores.items(), key=lambda x: x[1])[:2]
        
        # 전체 점수
        avg_score = sum(joint_scores.values()) / len(joint_scores)
        
        # 피드백 메시지 생성
        if avg_score >= 90:
            message = "훌륭한 자세입니다! 계속 유지하세요."
        elif avg_score >= 70:
            message = "좋은 자세입니다. "
            if worst_joints[0][1] < 70:
                # 관절 이름 한글화
                joint_name = self._translate_joint_name(worst_joints[0][0])
                message += f"{joint_name} 각도를 약간 조정하면 더 좋아질 것 같습니다."
        elif avg_score >= 50:
            message = "자세가 조금 부정확합니다. "
            for joint, score in worst_joints:
                if score < 70:
                    joint_name = self._translate_joint_name(joint)
                    message += f"{joint_name}의 각도를 조정해 보세요. "
        else:
            message = "자세가 많이 부정확합니다. 다음 부분을 조정해 보세요: "
            for joint, score in worst_joints:
                joint_name = self._translate_joint_name(joint)
                message += f"{joint_name}, "
            message = message[:-2] + "."
        
        return message
    
    def _translate_joint_name(self, joint_name: str) -> str:
        """
        관절 이름을 영어에서 한글로 변환
        
        Args:
            joint_name: 영어 관절 이름
            
        Returns:
            한글 관절 이름
        """
        translations = {
            "left_elbow": "왼쪽 팔꿈치",
            "right_elbow": "오른쪽 팔꿈치",
            "left_knee": "왼쪽 무릎",
            "right_knee": "오른쪽 무릎",
            "left_hip": "왼쪽 엉덩이",
            "right_hip": "오른쪽 엉덩이",
            "left_shoulder": "왼쪽 어깨",
            "right_shoulder": "오른쪽 어깨",
            "left_wrist": "왼쪽 손목",
            "right_wrist": "오른쪽 손목",
            "left_ankle": "왼쪽 발목",
            "right_ankle": "오른쪽 발목"
        }
        
        return translations.get(joint_name, joint_name)
    
    def calculate_distance_similarity(self, reference_keypoints: List[Dict], 
                                     current_keypoints: List[Dict]) -> Dict:
        """
        참조 포즈와 현재 포즈의 유사도를 L2 거리 기반으로 계산
        
        Args:
            reference_keypoints: 참조 포즈 키포인트
            current_keypoints: 현재 포즈 키포인트
            
        Returns:
            유사도 점수 및 세부 분석 결과
        """
        # 키포인트를 딕셔너리로 변환
        ref_kp = {kp["part"]: kp for kp in reference_keypoints}
        cur_kp = {kp["part"]: kp for kp in current_keypoints}
        
        # 관절별 거리 계산
        distances = {}
        valid_count = 0
        total_distance = 0
        
        for joint in self.key_joints:
            if (joint in ref_kp and joint in cur_kp and
                ref_kp[joint].get("x") is not None and ref_kp[joint].get("y") is not None and
                cur_kp[joint].get("x") is not None and cur_kp[joint].get("y") is not None and
                ref_kp[joint].get("confidence", 0) > 0.3 and cur_kp[joint].get("confidence", 0) > 0.3):
                
                # 관절 좌표 정규화 필요 (여기서는 단순화를 위해 생략)
                
                # 유클리드 거리 계산
                dist = math.sqrt(
                    (ref_kp[joint]["x"] - cur_kp[joint]["x"])**2 + 
                    (ref_kp[joint]["y"] - cur_kp[joint]["y"])**2
                )
                
                distances[joint] = dist
                valid_count += 1
                total_distance += dist
        
        if valid_count == 0:
            return {
                "score": 0,
                "joint_distances": {},
                "message": "충분한 관절 정보를 찾을 수 없습니다."
            }
        
        # 평균 거리
        avg_distance = total_distance / valid_count
        
        # 관절별 점수 계산 (거리가 작을수록 높은 점수)
        joint_scores = {}
        max_allowed_distance = 100  # 픽셀 단위, 조정 가능
        
        for joint, dist in distances.items():
            # 거리가 max_allowed_distance 이상이면 0점, 0이면 100점
            score = max(0, 100 - (dist * 100 / max_allowed_distance))
            joint_scores[joint] = score
        
        # 전체 점수
        overall_score = sum(joint_scores.values()) / len(joint_scores)
        
        return {
            "score": overall_score,
            "joint_scores": joint_scores,
            "joint_distances": distances,
            "message": self._generate_distance_feedback(joint_scores, distances)
        }
    
    def _generate_distance_feedback(self, joint_scores: Dict[str, float], 
                                   distances: Dict[str, float]) -> str:
        """
        거리 기반 자세 분석 결과에 기반한 피드백 메시지 생성
        
        Args:
            joint_scores: 관절별 점수
            distances: 관절별 거리
            
        Returns:
            피드백 메시지
        """
        if not joint_scores:
            return "포즈를 분석할 수 없습니다. 더 나은 포즈를 취해주세요."
        
        # 점수가 가장 낮은 관절 찾기
        worst_joints = sorted(joint_scores.items(), key=lambda x: x[1])[:2]
        
        # 전체 점수
        avg_score = sum(joint_scores.values()) / len(joint_scores)
        
        # 피드백 메시지 생성
        if avg_score >= 90:
            message = "모범적인 자세입니다!"
        elif avg_score >= 70:
            message = "좋은 자세입니다. "
            if worst_joints[0][1] < 70:
                joint_name = self._translate_joint_name(worst_joints[0][0])
                message += f"{joint_name}의 위치를 약간 조정하면 더 좋아질 것 같습니다."
        elif avg_score >= 50:
            message = "자세가 조금 부정확합니다. "
            for joint, score in worst_joints:
                if score < 70:
                    joint_name = self._translate_joint_name(joint)
                    message += f"{joint_name}의 위치를 조정해 보세요. "
        else:
            message = "자세가 많이 부정확합니다. 다음 부분을 조정해 보세요: "
            for joint, score in worst_joints:
                joint_name = self._translate_joint_name(joint)
                message += f"{joint_name}, "
            message = message[:-2] + "."
        
        return message 