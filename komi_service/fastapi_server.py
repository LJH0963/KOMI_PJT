from fastapi import FastAPI, WebSocket, HTTPException, Depends, Body, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import time
import os
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import json
import mimetypes

# MIME 타입 등록
mimetypes.add_type("video/mp4", ".mp4")

# 영상 저장 경로 설정
VIDEO_STORAGE_PATH = os.environ.get("VIDEO_STORAGE_PATH", "./video")

# 데이터 디렉토리 설정
DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", "./data")

# 운동 관련 데이터
exercise_data = {
    "exercises": [
        {
            "id": "squat",
            "name": "스쿼트",
            "description": "기본 하체 운동",
            "guide_videos": {
                "front": "/squat/front.mp4",
                "side": "/squat/side.mp4"
            }
        },
        {
            "id": "pushup",
            "name": "푸시업",
            "description": "상체 근력 운동",
            "guide_videos": {
                "front": "/pushup/front.mp4",
                "side": "/pushup/side.mp4"
            }
        },
        {
            "id": "lunge",
            "name": "런지",
            "description": "하체 균형 운동",
            "guide_videos": {
                "front": "/lunge/front.mp4",
                "side": "/lunge/side.mp4"
            }
        }
    ]
}

from fastapi_websocket import (
    active_connections,
    camera_info,
    data_lock,
    app_state,
    lifespan_manager,
    handle_websocket_connection,
    handle_camera_websocket,
    handle_stream_websocket
)

app = FastAPI(lifespan=lifespan_manager)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 디렉토리 생성 (없는 경우)
if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY, exist_ok=True)

# 정적 파일 서빙 설정
app.mount("/data", StaticFiles(directory=DATA_DIRECTORY), name="data")

# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    uptime = datetime.now() - app_state["start_time"]
    return {
        "status": "healthy" if app_state["is_running"] else "shutting_down",
        "connected_cameras": len(camera_info),
        "active_websockets": len(active_connections),
        "uptime_seconds": uptime.total_seconds(),
        "uptime_formatted": str(uptime)
    }

# 서버 시간 엔드포인트
@app.get("/server_time")
async def get_server_time():
    """서버의 현재 시간 정보 제공"""
    now = datetime.now()
    return {
        "server_time": now.isoformat(),
        "timestamp": time.time()
    }

# 카메라 목록 조회 엔드포인트
@app.get("/cameras")
async def get_cameras():
    """등록된 카메라 목록 조회"""
    with data_lock:
        # 현재 연결된 카메라만 반환 (WebSocket이 있는 카메라)
        active_cameras = [
            camera_id for camera_id, info in camera_info.items()
            if "websocket" in info
        ]
    
    return {"cameras": active_cameras, "count": len(active_cameras)}

# --- 운동 관련 엔드포인트 ---

@app.get("/exercises")
async def get_exercises():
    """사용 가능한 운동 목록 조회"""
    return {"exercises": exercise_data["exercises"]}

# 카메라 켜기/끄기 엔드포인트
@app.post("/cameras/{camera_id}/power")
async def camera_power_control(
    camera_id: str,
    power_status: bool = Body(..., embed=True)
):
    """카메라 전원 제어 (켜기/끄기)"""
    with data_lock:
        if camera_id not in camera_info:
            raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다")
        
        if "websocket" not in camera_info[camera_id]:
            raise HTTPException(status_code=400, detail="카메라가 현재 연결되어 있지 않습니다")
        
        try:
            # 카메라 클라이언트에 명령 전송
            websocket = camera_info[camera_id]["websocket"]
            await websocket.send_json({
                "type": "power_control",
                "power": power_status
            })
            
            # 카메라 상태 업데이트
            camera_info[camera_id]["power_status"] = "on" if power_status else "off"
            
            return {
                "status": "success",
                "camera_id": camera_id,
                "power": "on" if power_status else "off",
                "message": f"카메라가 {'켜졌습니다' if power_status else '꺼졌습니다'}"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"카메라 제어 중 오류 발생: {str(e)}")

@app.get("/exercise/{exercise_id}")
async def get_exercise_detail(exercise_id: str):
    """특정 운동의 상세 정보 조회"""
    # 운동 ID로 운동 찾기
    exercise = None
    for exercise_item in exercise_data["exercises"]:
        if exercise_item["id"] == exercise_id:
            exercise = exercise_item
            break
    
    if not exercise:
        raise HTTPException(status_code=404, detail="운동을 찾을 수 없습니다")
    
    return exercise

# 사용자 업로드 영상 스트리밍 엔드포인트
@app.get("/uploaded_videos/{video_id}")
async def get_uploaded_video(video_id: str):
    """사용자가 업로드한 영상을 스트리밍하여 제공"""
    video_path = os.path.join(VIDEO_STORAGE_PATH, "uploads", f"{video_id}.mp4")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="업로드된 영상을 찾을 수 없습니다")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"user_video_{video_id}.mp4"
    )

# --- 정밀 분석 관련 엔드포인트 ---

@app.post("/videos/upload")
async def upload_exercise_video(
    video: UploadFile = File(...),
    exercise_id: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    """운동 영상 업로드 및 분석 요청"""
    # 업로드 디렉토리 생성
    upload_dir = os.path.join(VIDEO_STORAGE_PATH, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # 파일 ID 생성
    video_id = f"{int(time.time())}_{exercise_id}"
    file_path = os.path.join(upload_dir, f"{video_id}.mp4")
    
    # 파일 저장
    try:
        with open(file_path, "wb") as buffer:
            contents = await video.read()
            buffer.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류 발생: {str(e)}")
    
    return {
        "video_id": video_id,
        "exercise_id": exercise_id,
        "status": "uploaded",
        "message": "영상이 업로드되었습니다. 분석이 진행 중입니다.",
        "video_url": f"/uploaded_videos/{video_id}"
    }

@app.post("/analyze/pose")
async def analyze_pose(
    pose_data: dict = Body(...),
    exercise_id: str = Form(...),
    video_id: Optional[str] = Form(None)
):
    """사용자의 자세 데이터를 분석하여 결과 제공"""
    # 실제로는 자세 분석 알고리즘을 통해 사용자 자세 평가
    # LLM을 활용한 자세 분석 피드백 제공
    # pose_data에는 관절 좌표 데이터가 포함되어 있어야 함
    
    analysis_result = {
        "exercise_id": exercise_id,
        "score": 85,  # 예시 점수
        "feedback": ["무릎 각도가 너무 좁습니다", "등이 굽어있습니다"],
        "comparison": {
            "hip_angle": {"user": 80, "reference": 90, "diff": -10},
            "knee_angle": {"user": 100, "reference": 110, "diff": -10}
        }
    }
    
    if video_id:
        analysis_result["video_id"] = video_id
    
    return analysis_result

@app.get("/analysis/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """특정 분석 결과 조회"""
    # 실제로는 DB에서 해당 분석 ID의 결과를 조회
    
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "result": {
            "score": 87,
            "feedback": ["무릎 각도 개선됨", "등 자세 교정 필요"],
            "detailed_analysis": "..."
        }
    }


# 웹소켓 연결 엔드포인트
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결 처리"""
    await handle_websocket_connection(websocket)

# 웹캠 카메라용 WebSocket 엔드포인트
@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """웹캠 클라이언트의 WebSocket 연결 처리"""
    await handle_camera_websocket(websocket)

# WebSocket을 통한 이미지 스트리밍 엔드포인트
@app.websocket("/ws/stream/{camera_id}")
async def stream_camera(websocket: WebSocket, camera_id: str):
    """특정 카메라의 이미지를 WebSocket으로 스트리밍"""
    await handle_stream_websocket(websocket, camera_id)

# 서버 실행 (직접 실행 시)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 