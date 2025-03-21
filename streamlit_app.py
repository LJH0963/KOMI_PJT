# ... existing code ...

# 웹캠 시작 버튼 클릭 시 실행되는 함수
def start_webcam_stream():
    if st.session_state.webcam_running:
        return
    
    async def setup_and_run_webcam():
        # 타임아웃 값을 크게 설정 (30초)
        client = AsyncWebcamClient(timeout=30.0)
        # 응답 처리 콜백 설정
        client.on_response = process_pose_response
        
        # 연결 시도
        connection_attempts = 0
        max_attempts = 3
        
        while connection_attempts < max_attempts:
            try:
                await client.connect()
                break
            except Exception as e:
                connection_attempts += 1
                if connection_attempts >= max_attempts:
                    st.error(f"서버 연결 실패: {str(e)}")
                    st.session_state.webcam_running = False
                    return
                # 재시도 전 대기
                await asyncio.sleep(1)
        
        # 웹캠 시작 및 스트리밍
        await client.start_webcam()
        await client.run_stream()
    
    st.session_state.webcam_running = True
    st.session_state.webcam_client_thread = run_async_function(setup_and_run_webcam)

# 오류 발생 시 재시도 로직 추가
def handle_connection_error():
    if st.session_state.webcam_running:
        # 현재 실행 중인 스레드 정리
        if st.session_state.webcam_client_thread:
            # (참고: 파이썬 스레드는 강제 종료가 어려움)
            st.session_state.webcam_client_thread = None
        
        # 상태 초기화 및 재시작
        st.session_state.webcam_running = False
        start_webcam_stream()
        st.success("연결이 재설정되었습니다.")

# UI에 재연결 버튼 추가
if st.button("연결 재설정"):
    handle_connection_error()

# ... existing code ... 