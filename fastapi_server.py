# ... existing code ...

# 비동기 처리 최적화를 위한 설정
executor = ThreadPoolExecutor(max_workers=4)  # 동시 처리 작업 수 제한

# Pose Detection 비동기 함수 최적화
async def detect_pose_async(frame):
    loop = asyncio.get_event_loop()
    # 계산 비용이 높은 작업은 ThreadPoolExecutor로 처리
    return await loop.run_in_executor(executor, pose_detector.detect_pose, frame)

# WebSocket 처리 최적화
@app.websocket("/ws/pose_stream")
async def pose_stream_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        # 클라이언트별 처리 상태 추적
        processing = False
        
        while True:
            # 이전 처리가 진행 중이면 새 프레임 스킵 (프레임 드롭)
            if processing:
                # 비동기적으로 메시지를 받되 처리하지 않음
                await websocket.receive_text()
                continue
            
            # 프레임 처리 시작
            processing = True
            
            try:
                # 클라이언트로부터 프레임 데이터 수신
                data = await websocket.receive_text()
                frame_data = json.loads(data)
                
                # 프레임 디코딩 및 처리
                img_bytes = base64.b64decode(frame_data["frame"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 비동기로 포즈 감지 - 타임아웃 처리 추가
                try:
                    # 타임아웃 설정으로 너무 오래 걸리는 처리 방지
                    pose_results = await asyncio.wait_for(
                        detect_pose_async(frame), 
                        timeout=5.0  # 5초 타임아웃 (필요에 따라 조정)
                    )
                    
                    # 결과 전송
                    await websocket.send_json({
                        "pose_data": pose_results,
                        "accuracy": calculate_pose_accuracy(pose_results)
                    })
                except asyncio.TimeoutError:
                    # 처리 시간 초과 시 간단한 오류 메시지 반환
                    await websocket.send_json({
                        "status": "error", 
                        "message": "Processing timeout"
                    })
            finally:
                # 처리 완료 플래그 설정
                processing = False
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

# 비동기 라우터 핸들러 수정
@app.post("/process_frame")
async def process_frame(frame_data: FrameData):
    try:
        # base64 문자열을 이미지로 변환
        img_bytes = base64.b64decode(frame_data.frame)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 비동기로 얼굴 인식 처리
        faces_result = await detect_faces_async(frame)
        
        # 감정 분석은 얼굴이 있을 때만 수행
        emotions_result = None
        if faces_result and len(faces_result) > 0:
            emotions_result = await analyze_emotions_async(frame, faces_result)
        
        return {
            "status": "success",
            "faces": faces_result,
            "emotions": emotions_result
        }
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return {"status": "error", "message": str(e)}

# 얼굴 인식 비동기 함수
async def detect_faces_async(frame):
    # CPU 집약적 작업은 ThreadPoolExecutor로 처리
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, face_detector.detect_faces, frame)

# 감정 분석 비동기 함수
async def analyze_emotions_async(frame, faces):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, emotion_analyzer.analyze, frame, faces)

# 음성 처리 비동기 함수
@app.post("/process_audio")
async def process_audio(audio_data: AudioData):
    try:
        # base64 문자열을 오디오로 변환
        audio_bytes = base64.b64decode(audio_data.audio)
        
        # 비동기로 오디오 처리
        result = await process_audio_async(audio_bytes, audio_data.sample_rate)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return {"status": "error", "message": str(e)}

# 오디오 처리 비동기 함수
async def process_audio_async(audio_bytes, sample_rate):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, audio_processor.process, audio_bytes, sample_rate)

# 서버 설정 최적화
if __name__ == "__main__":
    import uvicorn
    
    # Uvicorn 서버 설정 최적화
    uvicorn.run(
        "fastapi_server:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        timeout_keep_alive=120,  # Keep-alive 타임아웃 설정
        workers=1  # 워커 수 (비동기 코드에는 일반적으로 1개 권장)
    ) 