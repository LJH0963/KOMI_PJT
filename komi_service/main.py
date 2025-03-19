import os
import uuid
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time
import json
from datetime import datetime
from contextlib import asynccontextmanager

# 모듈 임포트
from komi_service.modules.websocket_manager import WebSocketManager
from komi_service.modules.pose_estimation import detect_pose, compare_poses, get_guide_pose
from komi_service.modules.config import APP_SETTINGS, UPLOAD_DIR
from komi_service.modules.llm_integration import get_llm_analysis, get_exercise_recommendation

# 시작 이벤트
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 초기화 작업"""
    print("KOMI 서비스 시작 중...")
    
    # 업로드 디렉토리 확인
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"업로드 디렉토리 준비 완료: {UPLOAD_DIR}")
    
    print("KOMI 서비스가 준비되었습니다.")
    yield
    # 앱 종료 시 정리 작업
    print("KOMI 서비스를 종료합니다.")

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="KOMI 서비스 API",
    description="Korean Open Metadata Initiative 자세 분석 및 운동 추천 API",
    version="0.0.1",
    lifespan=lifespan
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 구체적인 오리진 지정 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 웹소켓 매니저 인스턴스
websocket_manager = WebSocketManager()

# 세션 데이터 저장소 (실제 구현에서는 데이터베이스 사용 권장)
session_data = {}

# 운동 유형 목록
EXERCISE_TYPES = [
    {"id": "shoulder", "name": "어깨 운동", "description": "어깨 스트레칭 및 강화 운동"},
    {"id": "knee", "name": "무릎 운동", "description": "무릎 관절 강화 운동"},
    {"id": "posture", "name": "자세 교정", "description": "바른 자세 교정 운동"}
]

@app.get("/", tags=["기본"])
async def root():
    """서비스 상태 확인"""
    return {
        "status": "success",
        "service": "KOMI API",
        "version": app.version,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/exercises", tags=["운동"])
async def list_exercises():
    """사용 가능한 운동 목록 반환"""
    return {"exercises": EXERCISE_TYPES}

@app.post("/pose/upload", tags=["포즈"])
async def upload_image(
    file: UploadFile = File(...),
    exercise_type: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """
    이미지 업로드 및 포즈 분석 엔드포인트
    
    - 이미지 업로드 및 저장
    - 포즈 감지 수행
    - 가이드 포즈와 비교하여 정확도 계산
    - 결과 반환 및 WebSocket으로 브로드캐스트
    """
    try:
        # 이미지 데이터 읽기
        image_data = await file.read()
        
        # 세션 ID 생성 또는 기존 ID 사용
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # 파일명 생성 및 저장 (실제 저장은 생략, 실제 구현에서는 저장 필요)
        timestamp = int(time.time())
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        file_name = f"{session_id}_{timestamp}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        # 파일 저장 (실제 구현에서는 필요)
        # with open(file_path, "wb") as f:
        #     f.write(image_data)
        
        # 포즈 감지 수행 (더미 함수 사용)
        pose_data = detect_pose(image_data)
        
        # 세션 데이터에 결과 추가
        if session_id not in session_data:
            session_data[session_id] = []
        
        # 가이드 포즈와 비교 (운동 유형이 지정된 경우)
        accuracy = 0
        similarity_details = {}
        
        if exercise_type and pose_data and "pose" in pose_data and pose_data["pose"]:
            accuracy, similarity_details = compare_poses(pose_data, exercise_type)
        
        # 세션 데이터 저장
        session_item = {
            "timestamp": datetime.now().isoformat(),
            "file_name": file_name,
            "type": "pose_analysis",
            "pose_data": pose_data,
            "exercise_type": exercise_type,
            "accuracy": accuracy
        }
        session_data[session_id].append(session_item)
        
        # WebSocket으로 포즈 데이터 브로드캐스트
        await websocket_manager.broadcast_pose_data(
            pose_data, 
            accuracy, 
            similarity_details
        )
        
        # 응답 생성
        response = {
            "session_id": session_id,
            "file_name": file_name,
            "pose_data": pose_data,
            "accuracy": accuracy,
            "similarity_details": similarity_details
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"이미지 업로드 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"이미지 처리 중 오류 발생: {str(e)}"}
        )

@app.websocket("/pose/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 엔드포인트"""
    await websocket_manager.connect(websocket)
    
    try:
        # 초기 운동 목록 전송
        await websocket_manager.send_exercise_list(EXERCISE_TYPES)
        
        # 클라이언트에서 메시지 수신 대기
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                # 메시지 유형에 따른 처리
                if message.get("type") == "select_exercise":
                    exercise_id = message.get("exercise_id")
                    print(f"운동 선택: {exercise_id}")
                    
                    # 가이드 포즈 데이터 전송 (실제 구현에서는 여기에 추가)
                    guide_pose = get_guide_pose(exercise_id)
                    await websocket_manager.send_personal_message({
                        "type": "guide_pose",
                        "exercise_id": exercise_id,
                        "guide_pose": guide_pose
                    }, websocket)
                    
            except json.JSONDecodeError:
                print(f"잘못된 JSON 형식: {data}")
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"웹소켓 오류: {str(e)}")
        websocket_manager.disconnect(websocket)

@app.get("/analysis/{session_id}", tags=["분석"])
async def get_session_analysis(session_id: str):
    """세션 데이터 분석 결과 제공"""
    if session_id not in session_data or not session_data[session_id]:
        return JSONResponse(
            status_code=404,
            content={"error": f"세션 {session_id}를 찾을 수 없습니다."}
        )
    
    try:
        # LLM 기반 분석 수행
        analysis = await get_llm_analysis(session_data[session_id])
        
        return {
            "session_id": session_id,
            "analysis": analysis,
            "frames_count": len(session_data[session_id])
        }
    
    except Exception as e:
        print(f"세션 분석 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"세션 분석 중 오류 발생: {str(e)}"}
        )

@app.post("/recommendations", tags=["추천"])
async def get_recommendations(
    medical_condition: str = Form(...),
    pain_level: int = Form(..., ge=1, le=10),
    previous_exercise: Optional[str] = Form(None)
):
    """의료 상태 기반 운동 추천 제공"""
    try:
        recommendations = await get_exercise_recommendation(
            medical_condition,
            pain_level,
            previous_exercise
        )
        
        return recommendations
        
    except Exception as e:
        print(f"추천 생성 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"추천 생성 중 오류 발생: {str(e)}"}
        )

@app.get("/guide/{exercise_type}", tags=["가이드"])
async def get_exercise_guide(exercise_type: str):
    """특정 운동의 가이드 포즈 데이터 제공"""
    guide_pose = get_guide_pose(exercise_type)
    
    if not guide_pose:
        return JSONResponse(
            status_code=404,
            content={"error": f"운동 유형 '{exercise_type}'에 대한 가이드 포즈가 없습니다."}
        )
    
    return {
        "exercise_type": exercise_type,
        "guide_pose": guide_pose
    }

@app.delete("/session/{session_id}", tags=["세션"])
async def delete_session(session_id: str):
    """세션 데이터 삭제"""
    if session_id in session_data:
        del session_data[session_id]
        return {"status": "success", "message": f"세션 {session_id} 삭제됨"}
    
    return JSONResponse(
        status_code=404,
        content={"error": f"세션 {session_id}를 찾을 수 없습니다."}
    )

if __name__ == "__main__":
    # 애플리케이션 실행
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=APP_SETTINGS["port"],
        reload=APP_SETTINGS["debug"]
    )