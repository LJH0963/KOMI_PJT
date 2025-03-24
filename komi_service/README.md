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
- Streamlit 사용자 인터페이스

## 주요 컴포넌트 역할

### 1. FastAPI 서버 (fastapi_server.py)
- **역할**: 백엔드 서버 및 WebSocket 통신 허브
- **주요 기능**:
  - WebSocket 연결 관리 (`/ws/camera`, `/ws/monitor` 엔드포인트)
  - 실시간 이미지 처리 및 포즈 감지
  - 카메라 클라이언트와 모니터 클라이언트 간 데이터 중계
  - 비동기 이미지 처리 및 메시지 브로드캐스트
  - RESTful API 엔드포인트 제공

#### API 엔드포인트 목록
| 엔드포인트 | 메소드 | 설명 |
|------------|--------|------|
| `/` | GET | 서버 상태 확인 |
| `/cameras` | GET | 연결된 카메라 목록 조회 |
| `/latest_image/{camera_id}` | GET | 특정 카메라의 최신 이미지와 포즈 데이터 JSON 형식으로 조회 |
| `/get-image/{camera_id}` | GET | 특정 카메라의 최신 이미지를 바이너리로 직접 조회 |
| `/ws/camera` | WebSocket | 카메라 클라이언트 연결 엔드포인트 |
| `/ws/monitor` | WebSocket | 모니터링 클라이언트 연결 엔드포인트 |

### 2. Streamlit 앱 (streamlit_app.py)
- **역할**: 사용자 인터페이스 및 모니터링 클라이언트
- **주요 기능**:
  - 연결된 카메라 목록 표시 및 선택
  - 실시간 카메라 피드 표시
  - 포즈 데이터 시각화
  - HTTP와 WebSocket을 통한 서버 연결
  - 비동기 이미지 처리 및 UI 업데이트

### 3. 웹캠 클라이언트 (webcam_client.py)
- **역할**: 카메라 영상 캡처 및 서버 전송
- **주요 기능**:
  - 로컬 카메라 접근 및 프레임 캡처
  - 이미지 압축 및 Base64 인코딩
  - WebSocket을 통한 서버 연결 및 데이터 전송
  - 다중 카메라 지원 및 자동 카메라 선택
  - FPS 제어 및 최적화된 전송

## 실행 방법

### 1. FastAPI WebSocket 서버 실행

기본 설정: 호스트 0.0.0.0, 포트 8000

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--host` | 서버 호스트 주소 | 0.0.0.0 |
| `--port` | 서버 포트 번호 | 8000 |
| `--debug` | 디버그 모드 활성화 | False |

```bash
uvicorn komi_service.fastapi_server:app --host 0.0.0.0 --port 8000
```

### 2. Streamlit 앱 실행
```bash
streamlit run komi_service/streamlit_app.py
```

Streamlit 앱은 기본적으로 웹 브라우저를 열고 http://localhost:8501 에서 실행됩니다.

### 3. 웹캠 클라이언트 실행
```bash
python komi_service/webcam_client.py --cameras camera_12345:0
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--server` | 서버 WebSocket URL | ws://localhost:8000/ws/camera |
| `--camera` | 사용할 카메라 인덱스 | 자동 선택 |
| `--fps` | 초당 프레임 수 | 10 |
| `--width` | 이미지 너비 | 160 |
| `--height` | 이미지 높이 | 120 |

## 실행 순서
1. 먼저 FastAPI 서버를 실행합니다.
2. Streamlit 앱을 실행합니다.
3. 웹캠 클라이언트를 실행합니다.

## 시스템 아키텍처

### WebSocket 기반 통신

KOMI 서비스는 FastAPI의 WebSocket을 활용한 비동기 통신 방식을 사용합니다. 이를 통해 실시간으로 웹캠 이미지를 서버에 전송하고 포즈 감지 결과를 받을 수 있습니다.

#### 주요 모듈
- `fastapi_server.py`: FastAPI 기반 WebSocket 서버
  - 웹캠 클라이언트 엔드포인트 (`/ws/camera`)
  - 모니터링 엔드포인트 (`/ws/monitor`)
  - 포즈 감지 기능

- `webcam_client.py`: WebSocket 웹캠 클라이언트
  - 실시간 카메라 캡처 및 전송
  - FPS 제어 및 서버 통신

- `streamlit_app.py`: Streamlit 기반 사용자 인터페이스
  - 카메라 영상 표시
  - 포즈 데이터 시각화
  - 실시간 피드백 표시

#### 통신 프로토콜

##### 웹캠 클라이언트 -> 서버
```json
{
  "type": "frame",
  "timestamp": 1616161616.123,
  "image_data": "base64로 인코딩된 이미지 데이터"
}
```

##### 서버 -> 웹캠 클라이언트
```json
{
  "type": "connection_successful",
  "camera_id": "camera_12345",
  "message": "서버에 연결되었습니다."
}
```

또는 이미지 처리 결과:

```json
{
  "type": "frame_processed",
  "client_id": "camera_12345",
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
  "type": "status_update",
  "camera_id": "camera_12345",
  "timestamp": 1616161616.234,
  "has_new_image": true,
  "pose_data": {
    // 포즈 데이터
  }
}
```

또는 카메라 목록:

```json
{
  "type": "cameras_list",
  "cameras": ["camera_12345", "camera_67890"]
}
```

#### 데이터 흐름
1. 웹캠 클라이언트가 카메라 영상을 캡처합니다.
2. 캡처된 프레임을 Base64로 인코딩하여 WebSocket으로 서버에 전송합니다.
3. 서버는 프레임을 수신하여 이미지를 디코딩하고 포즈 감지를 수행합니다.
4. 처리 결과를 원본 클라이언트에게 응답하고 모니터링 클라이언트에게 브로드캐스트합니다.
5. Streamlit 앱은 서버에서 처리된 이미지와 포즈 데이터를 받아 사용자에게 표시합니다.

## 개발 정보

### 필요 라이브러리
- FastAPI: WebSocket 서버 구현
- uvicorn: ASGI 서버
- OpenCV: 이미지 처리 및 웹캠 접근
- websocket-client: Python WebSocket 클라이언트
- NumPy: 배열 처리
- aiohttp: 비동기 HTTP 클라이언트
- Streamlit: 사용자 인터페이스
- PIL: 이미지 처리

### 확장 가능성
- 복수의 웹캠 클라이언트 동시 연결 지원
- 포즈 감지 모델 교체 및 업그레이드
- 실시간 분석 결과 시각화
- 모바일 앱 클라이언트 개발
- LLM 기반 의료 지식 응답 시스템 통합

### 비동기 처리
KOMI 서비스는 asyncio와 WebSocket을 활용한 비동기 처리를 통해 실시간 포즈 감지와 분석을 수행합니다. 이를 통해 여러 클라이언트의 동시 연결과 효율적인 리소스 관리가 가능합니다.


#### 다중 카메라 테스트
```
uvicorn komi_service.fastapi_server:app --host 0.0.0.0 --port 8000
streamlit run komi_service/streamlit_app.py -- --server_url http://192.168.10.87:8000
python komi_service/webcam_client.py --cameras camera_12346:0 --server http://192.168.10.87:8000
```


# TODO:
- !! 수정된 코드 작동 확인 !!
- 싱크 맞추기(멀티 카메라 동기화)
- 카메라 클라이언트에서 좌표 정보 받기
- 스트림릿에 점수 출력하기
