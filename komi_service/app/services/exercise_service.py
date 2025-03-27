"""
운동 세션 관리 서비스
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import HTTPException

from app.models.exercise import ExerciseModel
from app.config import EXERCISE_DATA

class ExerciseService:
    """운동 세션 관리 서비스"""
    
    @staticmethod
    def get_all_exercises() -> Dict[str, Any]:
        """사용 가능한 모든 운동 목록 반환
        
        Returns:
            운동 목록 데이터
        """
        return EXERCISE_DATA
    
    @staticmethod
    def find_exercise_by_id(exercise_id: str) -> Optional[Dict[str, Any]]:
        """ID로 운동 정보 조회
        
        Args:
            exercise_id: 운동 ID
            
        Returns:
            운동 정보 또는 None
        """
        for exercise in EXERCISE_DATA["exercises"]:
            if exercise["id"] == exercise_id:
                return exercise
        return None
    
    @staticmethod
    def validate_exercise_id(exercise_id: str) -> None:
        """운동 ID 유효성 검증
        
        Args:
            exercise_id: 운동 ID
            
        Raises:
            HTTPException: 유효하지 않은 운동 ID
        """
        if not ExerciseService.find_exercise_by_id(exercise_id):
            raise HTTPException(status_code=404, detail="운동을 찾을 수 없습니다")
    
    @staticmethod
    def create_exercise_session(exercise_id: str, camera_ids: List[str]) -> Dict[str, Any]:
        """운동 세션 생성
        
        Args:
            exercise_id: 운동 ID
            camera_ids: 카메라 ID 목록
            
        Returns:
            생성된 세션 정보
        """
        # 운동 ID 유효성 검증
        ExerciseService.validate_exercise_id(exercise_id)
        
        # 세션 생성
        session_id = ExerciseModel.create_session(exercise_id, camera_ids)
        
        return {
            "session_id": session_id,
            "status": "ready"
        }
    
    @staticmethod
    def get_exercise_session(session_id: str) -> Dict[str, Any]:
        """세션 정보 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            세션 정보
            
        Raises:
            HTTPException: 세션을 찾을 수 없음
        """
        session = ExerciseModel.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        return session
    
    @staticmethod
    def end_exercise_session(session_id: str) -> Dict[str, Any]:
        """세션 종료
        
        Args:
            session_id: 세션 ID
            
        Returns:
            세션 정보
            
        Raises:
            HTTPException: 세션을 찾을 수 없음
        """
        # 세션 종료
        if not ExerciseModel.end_session(session_id):
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        return {
            "session_id": session_id,
            "status": "completed"
        } 