"""
설정 관리 모듈
환경 변수, 서버 설정 등 관리
"""

import os
from pathlib import Path

# 기본 디렉터리 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "komi_service", "data")

# 서버 설정
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# 웹소켓 설정
PING_INTERVAL = 30  # 30초마다 핑 전송
MAX_IDLE_TIME = 60  # 60초 동안 응답이 없으면 연결 종료
CLEANUP_INTERVAL = 60  # 60초마다 연결 정리
CAMERA_PING_INTERVAL = 15  # 카메라 클라이언트 핑 간격

# 데이터 디렉터리 생성
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

# 운동 데이터 설정
EXERCISE_DATA = {
    "exercises": [
        {
            "id": "squat",
            "name": "스쿼트",
            "description": "기본적인 하체 운동",
            "guide_videos": {
                "front": "/squat/front.mp4",
                "side": "/squat/side.mp4"
            },
            "difficulty": "초급"
        },
        {
            "id": "pushup",
            "name": "푸시업",
            "description": "상체 근력 운동",
            "guide_videos": {
                "front": "/pushup/front.mp4",
                "side": "/pushup/side.mp4"
            },
            "difficulty": "중급"
        },
        {
            "id": "lunge",
            "name": "런지",
            "description": "하체 균형 운동",
            "guide_videos": {
                "front": "/lunge/front.mp4",
                "side": "/lunge/side.mp4"
            },
            "difficulty": "중급"
        }
    ]
} 
