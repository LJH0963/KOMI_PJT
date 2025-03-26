import os
from typing import Dict, Any

# 환경 변수에서 설정 로드
DEBUG = os.environ.get("KOMI_DEBUG", "True").lower() in ("true", "1", "t")
PORT = int(os.environ.get("KOMI_PORT", "8001"))
UPLOAD_DIR = os.environ.get("KOMI_UPLOAD_DIR", "uploads")

# YOLO 모델 설정 (실제 구현에서는 진짜 모델 로드)
class DummyModel:
    """더미 YOLO 모델 클래스"""
    def __init__(self):
        print("더미 YOLO 모델 초기화")
        
    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """더미 예측 함수"""
        return {"keypoints": [], "boxes": []}

# 더미 YOLO 모델 생성
yolo_model = DummyModel()

# 애플리케이션 설정
APP_SETTINGS = {
    "debug": DEBUG,
    "port": PORT,
    "model_type": "dummy",
    "upload_dir": UPLOAD_DIR
}

# 업로드 디렉토리 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)
print(f"업로드 디렉토리 확인: {UPLOAD_DIR}")

print(f"KOMI 서비스 설정 로드 완료: {APP_SETTINGS}")
