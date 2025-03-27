"""
운동 세션 모델 및 상태 관리
"""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

# 운동 세션 저장소
exercise_sessions: Dict[str, dict] = {}

class ExerciseSessionCreate(BaseModel):
    """운동 세션 생성 요청 모델"""
    exercise_id: str
    camera_ids: List[str]

class ExerciseSession(BaseModel):
    """운동 세션 모델"""
    session_id: str
    exercise_id: str
    camera_ids: List[str]
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None

class ExerciseModel:
    """운동 세션 관련 데이터 모델"""
    
    @staticmethod
    def create_session(exercise_id: str, camera_ids: List[str]) -> str:
        """새 운동 세션 생성"""
        session_id = f"session_{len(exercise_sessions) + 1}"
        exercise_sessions[session_id] = {
            "exercise_id": exercise_id,
            "camera_ids": camera_ids,
            "status": "ready",
            "start_time": datetime.now(),
            "end_time": None
        }
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> Optional[dict]:
        """세션 정보 조회"""
        if session_id in exercise_sessions:
            return exercise_sessions[session_id]
        return None
    
    @staticmethod
    def end_session(session_id: str) -> bool:
        """세션 종료"""
        if session_id in exercise_sessions:
            exercise_sessions[session_id]["status"] = "completed"
            exercise_sessions[session_id]["end_time"] = datetime.now()
            return True
        return False
    
    @staticmethod
    def get_all_sessions() -> List[dict]:
        """모든 세션 목록 반환"""
        return list(exercise_sessions.values()) 