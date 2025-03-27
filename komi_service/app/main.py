"""
KOMI 서비스 FastAPI 메인 애플리케이션
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import DATA_DIR
from app.api import health, camera, exercise, video
from app.websockets import camera as ws_camera, stream as ws_stream, pose as ws_pose
from app.websockets.manager import websocket_manager

# 애플리케이션 생명주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행할 코드"""
    # 시작 시 실행
    
    # 웹소켓 관리자 정리 태스크 시작
    cleanup_task = asyncio.create_task(websocket_manager.periodic_cleanup())
    
    # 컨텍스트 전환
    yield
    
    # 종료 시 실행
    
    # 정리 태스크 취소
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

# FastAPI 앱 생성
app = FastAPI(
    title="KOMI Service API",
    description="Kinematic Optimization & Motion Intelligence",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# API 라우터 등록
app.include_router(health.router)
app.include_router(camera.router)
app.include_router(exercise.router)
app.include_router(video.router)

# 웹소켓 라우터 등록
app.include_router(ws_camera.router)
app.include_router(ws_stream.router)
app.include_router(ws_pose.router)

# 루트 엔드포인트
@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "name": "KOMI Service API",
        "version": "0.1.0",
        "status": "running"
    }

# 서버 실행 (직접 실행 시)
if __name__ == "__main__":
    import uvicorn
    from app.config import DEFAULT_HOST, DEFAULT_PORT
    
    uvicorn.run(
        "app.main:app", 
        host=DEFAULT_HOST, 
        port=DEFAULT_PORT,
        reload=True
    ) 