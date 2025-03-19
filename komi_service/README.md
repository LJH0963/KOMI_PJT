# KOMI 서비스

KOMI(Kinematic Optimization & Motion Intelligence) 서비스는 AI 기반 실시간 자세 분석 및 피드백 시스템입니다. 이 서비스는 사용자의 자세를 분석하고, 개인화된 운동 추천을 제공합니다.

## 프로젝트 구조

```
komi_service/
├── modules/                 # 주요 모듈
│   ├── pose_estimation.py   # 포즈 감지 및 분석 모듈
│   ├── llm_integration.py   # LLM 통합 모듈
│   ├── websocket_manager.py # 웹소켓 관리 모듈
│   ├── config.py            # 설정 모듈
│   └── __init__.py
├── docs/                    # 문서
├── main.py                  # FastAPI 서버 메인 파일
├── streamlit_app.py         # Streamlit 프론트엔드
└── requirements.txt         # 필수 패키지
```


## 실행 방법
    1. FastAPI 서버와 Streamlit 앱을 모두 실행합니다.
    2. 웹 브라우저에서 Streamlit UI 접속 (기본 포트: 8501)
    3. 카메라를 활성화하고 운동 유형을 선택합니다.
    4. 선택한 운동에 따라 실시간 자세 분석 및 피드백을 확인합니다.
    5. 세션 기록 기능을 활용해 운동 결과를 저장하고 확인할 수 있습니다.

### FastAPI 서버 실행
```bash
uvicorn komi_service.main:app --host 0.0.0.0 --port 8001 --reload
```

### Streamlit 앱 실행
```bash
streamlit run komi_service/streamlit_app.py
```

## API 엔드포인트

### 기본 API
- `GET /` : 서비스 상태 확인
- `GET /exercises` : 사용 가능한 운동 목록 반환

### 포즈 분석 API
- `POST /pose/upload` : 이미지 업로드 및 포즈 분석
- `WebSocket /pose/ws` : 실시간 포즈 데이터 스트림

### 분석 API
- `GET /analysis/{session_id}` : 세션 데이터 분석 결과 제공
- `POST /recommendations` : 의료 상태 기반 운동 추천 제공
- `GET /guide/{exercise_type}` : 운동 가이드 포즈 제공
- `DELETE /session/{session_id}` : 세션 데이터 삭제


