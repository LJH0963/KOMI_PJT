"""
운동 관련 API 라우터
"""

from fastapi import APIRouter, HTTPException
from typing import List

from app.services.exercise_service import ExerciseService
from app.models.exercise import ExerciseSessionCreate

router = APIRouter(tags=["exercises"])

@router.get("/exercises")
async def get_exercises():
    """사용 가능한 운동 목록 조회"""
    return ExerciseService.get_all_exercises()

@router.post("/exercise/start")
async def start_exercise(exercise_session: ExerciseSessionCreate):
    """운동 세션 시작"""
    return ExerciseService.create_exercise_session(
        exercise_session.exercise_id, 
        exercise_session.camera_ids
    )

@router.get("/exercise/{session_id}")
async def get_exercise_status(session_id: str):
    """운동 세션 상태 조회"""
    return ExerciseService.get_exercise_session(session_id)

@router.post("/exercise/{session_id}/end")
async def end_exercise(session_id: str):
    """운동 세션 종료"""
    return ExerciseService.end_exercise_session(session_id) 