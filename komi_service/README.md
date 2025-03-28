# KOMI 서비스

## 프로젝트 개요
KOMI(Kinematic Optimization & Motion Intelligence) 서비스는 포즈 감지 기술과 인공지능을 활용하여 사용자의 운동 자세를 분석하고 피드백을 제공하는 시스템입니다. AI 기반 실시간 관절 분석 및 LLM을 활용한 원격 진단 및 맞춤형 재활 운동 추천 기능을 제공합니다.

## 주요 기능
- 실시간 포즈 감지 및 분석 (YOLO11 기반)
- 운동 정확도 측정 및 시각적 피드백
- AI 기반 자세 평가 및 피드백
- LLM을 활용한 의료 상담 및 재활 운동 추천
- 다중 카메라 동기화를 통한 정확한 움직임 분석

## 시스템 구성

### 핵심 구성요소
- FastAPI 백엔드 서버 (WebSocket 지원)
- YOLO11 기반 포즈 감지 엔진
- Ollama 기반 LLM 모듈
- 웹캠 클라이언트-서버 통신 모듈
- Streamlit 사용자 인터페이스

### 주요 모듈 구성
```
komi_service/
├── fastapi_server.py  # WebSocket 서버 및 API 엔드포인트
├── webcam_client.py   # 카메라 캡처 및 스트리밍 클라이언트
├── streamlit_app.py   # 웹 기반 모니터링 인터페이스
├── pose_detection/    # 포즈 감지 모듈 (개발 예정)
│   ├── yolo_model.py     # YOLO11 포즈 감지 모델
│   └── pose_analyzer.py  # 자세 분석 알고리즘
└── llm/               # LLM 모듈 (개발 예정)
    ├── ollama_client.py  # Ollama 기반 LLM 통신
    └── medical_advisor.py # 의료 상담 및 운동 추천 엔진
```

## 주요 컴포넌트 역할

### 1. FastAPI 서버 (fastapi_server.py)
- **역할**: 백엔드 서버 및 WebSocket 통신 허브
- **주요 기능**:
  - WebSocket 연결 관리 (`/ws/camera`, `/ws/stream/{camera_id}` 엔드포인트)
  - 실시간 이미지 저장 및 중계
  - 카메라 클라이언트에서 수신한 이미지를 모니터링 클라이언트로 브로드캐스트
  - 비동기 연결 관리 및 메시지 처리
  - 포즈 감지 모델 호출 및 LLM 연동

#### API 엔드포인트 목록
| 엔드포인트 | 메소드 | 설명 |
|------------|--------|------|
| `/health` | GET | 서버 상태 및 연결 정보 확인 |
| `/server_time` | GET | 서버 시간 정보 (타임스탬프 동기화용) |
| `/cameras` | GET | 연결된 카메라 목록 조회 |
| `/ws/camera` | WebSocket | 카메라 클라이언트 연결 엔드포인트 |
| `/ws/stream/{camera_id}` | WebSocket | 특정 카메라 스트림을 구독하는 엔드포인트 |
| `/ws/updates` | WebSocket | 상태 업데이트 수신 엔드포인트 |
| `/analyze_pose` | POST | 포즈 분석 요청 (개발 예정) |
| `/exercise_feedback` | POST | 운동 피드백 요청 (개발 예정) |
| `/llm/advice` | POST | LLM 기반 의료 조언 요청 (개발 예정) |

### 2. Streamlit 앱 (streamlit_app.py)
- **역할**: 사용자 인터페이스 및 모니터링 클라이언트
- **주요 기능**:
  - 연결된 카메라 목록 표시 및 선택
  - 실시간 카메라 피드 표시 (최대 2대 동시 모니터링)
  - 타임스탬프 기반 카메라 동기화 시각화
  - HTTP와 WebSocket을 통한 서버 연결
  - 비동기 이미지 처리 및 UI 업데이트
  - 향후 기능: 포즈 감지 결과 시각화 및 운동 가이드 표시
  - 향후 기능: LLM 기반 의료 상담 및 운동 추천 UI

### 3. 웹캠 클라이언트 (webcam_client.py)
- **역할**: 카메라 영상 캡처 및 서버 전송
- **주요 기능**:
  - 로컬 카메라 접근 및 프레임 캡처
  - 서버 시간과의 동기화를 통한 정확한 타임스탬프 생성
  - 이미지 압축 및 Base64 인코딩을 통한 효율적인 전송
  - WebSocket을 통한 서버 연결 및 실시간 데이터 전송
  - 다중 카메라 지원 (카메라별 독립 스레드)
  - FPS 제어 및 자동 재연결 메커니즘

### 4. 포즈 감지 모듈
- **역할**: 이미지 내 사용자 포즈 감지 및 분석
- **주요 기능**:
  - YOLO11 기반 실시간 포즈 감지
  - 키포인트 추출 및 자세 분석
  - 자세 비교 알고리즘 (cosine similarity / angle based evaluation)
  - 자세 정확도 점수 계산

### 5. LLM 모듈
- **역할**: 의료 지식 기반 상담 및 운동 추천
- **주요 기능**:
  - Ollama 기반 자체 LLM 구축
  - 사용자 자세 데이터 기반 맞춤형 피드백 생성
  - RAG를 활용한 자세 교정 가이드 및 의료 조언 제공

## 실행 방법

### 1. FastAPI 서버 실행

```bash
uvicorn komi_service.fastapi_server:app --host 0.0.0.0 --port 8000
```


### 2. 웹캠 클라이언트 실행
```bash
python komi_service/webcam_client.py --cameras camera_12346:0 --server "http://localhost:8000" --fps 15 --quality 85 --max-width 640
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--cameras` | "카메라ID:인덱스" 형식으로 연결할 카메라 지정 | 필수 |
| `--server` | 서버 URL | http://localhost:8000 |
| `--quality` | 이미지 압축 품질 (0-100) | 85 |
| `--max-width` | 이미지 최대 폭 (픽셀) | 640 |
| `--fps` | 카메라 프레임 레이트 | 15 |

### 3. Streamlit 앱 실행
```bash
streamlit run komi_service/streamlit_app.py
```

Streamlit 앱은 기본적으로 웹 브라우저를 열고 http://localhost:8501 에서 실행됩니다.

## 실행 순서
1. 먼저 FastAPI 서버를 실행합니다.
2. 웹캠 클라이언트를 실행합니다.
3. Streamlit 앱을 실행합니다.

## 시스템 아키텍처

### WebSocket 기반 통신

KOMI 서비스는 FastAPI의 WebSocket을 활용한 비동기 통신 방식을 사용합니다. 이를 통해 실시간으로 웹캠 이미지 및 포즈 정보를 서버에 전송하고 분석 결과를 출력합니다.

### 이미지 처리 및 전송
- OpenCV를 활용한 카메라 캡처 및 이미지 처리
- JPEG 압축 (품질 조절 가능)
- Base64 인코딩을 통한 효율적인 이미지 전송
- 이미지 크기 제한 기능을 통한 네트워크 부하 최적화

### 연결 관리
- WebSocket을 활용한 양방향 실시간 통신
- 주기적인 핑/퐁 메커니즘을 통한 연결 상태 유지
- 지수 백오프 알고리즘을 통한 재연결 및 서버 부하 분산
- 연결 끊김 감지 및 자동 복구

### 포즈 감지 및 분석
- YOLO11 기반 실시간 포즈 감지
- 17개 주요 관절 키포인트 추출
- 자세 비교 알고리즘 구현
  - L2 Distance: 관절 좌표 간 거리 계산
  - Cosine Similarity: 관절 벡터 간 각도 유사도 분석
  - Dynamic Time Warping: 시간 흐름에 따른 동작 패턴 비교
- 사용자 자세 정확도 점수화 (0-100%)

### LLM 기반 의료 상담
- Ollama 기반 로컬 LLM 연동
- 의료 및 재활 관련 데이터 학습
- 실시간 분석 결과에 기반한 맞춤형 피드백
- 재활 운동 추천 및 자세 교정 가이드 제공

## 데이터 흐름

1. **데이터 수집**: 
   - 웹캠 클라이언트가 2대의 카메라에서 동시에 영상을 캡처
   - 서버 시간과 동기화하여 정확한 타임스탬프 부여
   - 이미지 압축 및 Base64 인코딩 후 WebSocket을 통해 서버로 전송

2. **서버 처리**:
   - 웹캠 이미지 수신 및 디코딩
   - 포즈 감지 모듈을 통한 관절 키포인트 추출
   - 자세 분석 알고리즘을 통한 정확도 평가
   - 분석 결과 데이터 생성

3. **데이터 전달**:
   - 처리된 이미지와 분석 결과를 Streamlit 클라이언트에 전송
   - WebSocket을 통한 실시간 데이터 스트리밍

4. **피드백 생성**:
   - 분석 결과에 기반한 사용자 피드백 생성
   - LLM을 활용한 자세 교정 가이드 및 운동 추천
   - 시각적 피드백 및 텍스트 조언 제공

5. **결과 표시**:
   - Streamlit 인터페이스를 통해 사용자에게 실시간 피드백 표시
   - 자세 정확도 점수 및 교정 가이드 시각화
   - 운동 가이드 영상과 비교 분석 제공

## 개발 정보

### 필요 라이브러리
- FastAPI: WebSocket 서버 구현
- uvicorn: ASGI 서버
- OpenCV: 이미지 처리 및 웹캠 접근
- NumPy: 배열 처리
- aiohttp: 비동기 HTTP 및 WebSocket 클라이언트
- Streamlit: 사용자 인터페이스
- PIL: 이미지 처리
- asyncio: 비동기 처리
- YOLO11: 포즈 감지 모델
- Ollama: 로컬 LLM 구동

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
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
streamlit run streamlit_app.py -- --server_url http://192.168.10.87:8000
python webcam_client.py --cameras camera_12346:0 --server http://192.168.10.87:8000
```

# TODO:
- !!중간에 카메라 연결이 끊겼을 때 새로고침을 하지 않아도 연동 되도록 처리 필요!!
- YOLO11 기반 포즈 감지 모듈 구현
- 자세 분석 알고리즘 개발 (cosine similarity / angle based evaluation)
- 운동 가이드 영상과 사용자 자세 비교 기능 추가
- Ollama 기반 LLM 연동 및 분석 결과에 기반한 맞춤형 피드백 기능 구현
- 재활 운동 추천 및 자세 교정 가이드 제공 기능 개발
