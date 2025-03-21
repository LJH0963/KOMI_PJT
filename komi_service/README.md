# KOMI 서비스

## 프로젝트 개요
KOMI(Korean Open Metadata Initiative) 서비스는 포즈 감지 기술과 인공지능을 활용하여 사용자의 운동 자세를 분석하고 피드백을 제공하는 시스템입니다.

## 주요 기능
- 실시간 포즈 감지 및 분석
- 운동 정확도 측정
- 다양한 운동 종류 지원
- AI 기반 운동 추천 및 분석

## 시스템 구성
- FastAPI 백엔드 서버
- WebSocket 기반 실시간 통신
- 웹캠 클라이언트-서버 통신

## 설치 방법
1. 저장소 클론
```bash
git clone https://github.com/yourusername/komi_service.git
cd komi_service
```

2. 가상환경 생성 (선택사항)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

### 1. FastAPI WebSocket 서버 실행
```bash
python -m komi_service.fastapi_server
```

기본 설정: 호스트 0.0.0.0, 포트 8000

#### 명령행 옵션
```bash
python -m komi_service.fastapi_server --host 127.0.0.1 --port 8080 --debug
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--host` | 서버 호스트 주소 | 0.0.0.0 |
| `--port` | 서버 포트 번호 | 8000 |
| `--debug` | 디버그 모드 활성화 | False |

### 2. 웹캠 클라이언트 실행
```bash
python -m komi_service.webcam_client
```

#### 명령행 옵션
```bash
python -m komi_service.webcam_client [camera_index] [server_url]
```

| 인수 | 설명 | 기본값 |
|------|------|--------|
| `camera_index` | 사용할 카메라 인덱스 | 사용자 선택 |
| `server_url` | 서버 WebSocket URL | ws://localhost:8000/ws |

## 시스템 아키텍처

### WebSocket 기반 통신

KOMI 서비스는 FastAPI의 WebSocket을 활용한 비동기 통신 방식을 사용합니다. 이를 통해 실시간으로 웹캠 이미지를 서버에 전송하고 포즈 감지 결과를 받을 수 있습니다.

#### 주요 모듈
- `fastapi_server.py`: FastAPI 기반 WebSocket 서버
  - 웹캠 클라이언트 엔드포인트 (`/ws`)
  - 모니터링 엔드포인트 (`/ws/monitor`)
  - 포즈 감지 기능

- `webcam_client.py`: WebSocket 웹캠 클라이언트
  - 실시간 카메라 캡처 및 전송
  - FPS 제어 및 서버 통신

#### 통신 프로토콜

##### 웹캠 클라이언트 -> 서버
```json
{
  "type": "webcam_frame",
  "timestamp": 1616161616.123,
  "data": "base64로 인코딩된 이미지 데이터"
}
```

##### 서버 -> 웹캠 클라이언트
```json
{
  "type": "frame_processed",
  "client_id": "client_12345",
  "timestamp": 1616161616.234,
  "pose_data": {
    "pose": [
      {
        "keypoints": [
          {"id": 0, "x": 100, "y": 200, "confidence": 0.9},
          // ... 기타 키포인트
        ]
      }
    ],
    "timestamp": 1616161616.234
  }
}
```

##### 서버 -> 모니터링 클라이언트
```json
{
  "type": "pose_update",
  "client_id": "client_12345",
  "timestamp": 1616161616.234,
  "pose_data": {
    // 포즈 데이터
  }
}
```

#### 데이터 흐름
1. 웹캠 클라이언트가 카메라 영상을 캡처합니다.
2. 캡처된 프레임을 Base64로 인코딩하여 WebSocket으로 서버에 전송합니다.
3. 서버는 프레임을 수신하여 이미지를 디코딩하고 포즈 감지를 수행합니다.
4. 처리 결과를 원본 클라이언트에게 응답하고 모니터링 클라이언트에게 브로드캐스트합니다.

## 개발 정보

### 필요 라이브러리
- FastAPI: WebSocket 서버 구현
- uvicorn: ASGI 서버
- OpenCV: 이미지 처리 및 웹캠 접근
- websocket-client: Python WebSocket 클라이언트
- NumPy: 배열 처리

### 확장 가능성
- 복수의 웹캠 클라이언트 동시 연결 지원
- 포즈 감지 모델 교체 및 업그레이드
- 실시간 분석 결과 시각화
- 모바일 앱 클라이언트 개발

