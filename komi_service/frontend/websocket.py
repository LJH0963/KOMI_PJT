import asyncio
import aiohttp
import json
import time
import random
import queue
from datetime import datetime
import threading

from frontend import utils
from frontend import config
from frontend.session import app_state
from frontend.camera import update_connection_status, init_sync_buffer

# 백그라운드 스레드에서 비동기 루프 실행
def run_async_loop():
    """비동기 루프를 실행하는 스레드 함수"""
    # 이 스레드 전용 이벤트 루프 생성
    loop = utils.get_event_loop()
    
    try:
        # 이미지 업데이트 태스크 생성
        task = loop.create_task(update_images())
        
        # 이벤트 루프 실행
        loop.run_until_complete(task)
    except Exception as e:
        print(f"비동기 루프 오류: {str(e)}")
    finally:
        # 모든 태스크 취소
        for task in asyncio.all_tasks(loop):
            task.cancel()
        
        # 취소된 태스크 완료 대기
        if asyncio.all_tasks(loop):
            loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
        
        # 루프 중지
        loop.stop()

# 비동기 이미지 업데이트 함수
async def update_images():
    """백그라운드에서 이미지를 가져오는 함수 (WebSocket 기반)"""
    # 세션 초기화
    await utils.init_session()
    
    # 초기 서버 시간 동기화
    app_state.server_time_offset, app_state.last_time_sync = await utils.sync_server_time(
        app_state.server_time_offset, app_state.last_time_sync
    )
    
    # 카메라별 WebSocket 연결 태스크 저장
    stream_tasks = {}
    pose_stream_tasks = {}  # 포즈 스트림 태스크
    
    try:
        while app_state.is_running:
            # 주기적 서버 시간 동기화 - 현재 시간과 비교하여 판단
            if time.time() - app_state.last_time_sync >= config.TIME_SYNC_INTERVAL:
                app_state.server_time_offset, app_state.last_time_sync = await utils.sync_server_time(
                    app_state.server_time_offset, app_state.last_time_sync
                )
            
            # 전역 변수로 카메라 ID 목록 확인
            camera_ids = app_state.selected_cameras
            
            if camera_ids:
                # 동기화 버퍼 초기화
                init_sync_buffer(camera_ids)
                
                # 기존 스트림 중 필요없는 것 종료
                for camera_id in list(stream_tasks.keys()):
                    if camera_id not in camera_ids:
                        if not stream_tasks[camera_id].done():
                            stream_tasks[camera_id].cancel()
                        del stream_tasks[camera_id]
                
                # 포즈 스트림 중 필요없는 것 종료
                for camera_id in list(pose_stream_tasks.keys()):
                    if camera_id not in camera_ids:
                        if not pose_stream_tasks[camera_id].done():
                            pose_stream_tasks[camera_id].cancel()
                        del pose_stream_tasks[camera_id]
                
                # 새 카메라에 대한 WebSocket 스트림 시작
                for camera_id in camera_ids:
                    # 이미지 스트림
                    if camera_id not in stream_tasks or stream_tasks[camera_id].done():
                        # 재연결 시도할 때 약간의 지연 추가 (무작위)
                        jitter = random.uniform(0, 0.5)
                        await asyncio.sleep(jitter)
                        stream_tasks[camera_id] = asyncio.create_task(
                            connect_to_camera_stream(camera_id)
                        )
                    
                    # 포즈 데이터 스트림
                    if camera_id not in pose_stream_tasks or pose_stream_tasks[camera_id].done():
                        # 재연결 시도할 때 약간의 지연 추가 (무작위)
                        jitter = random.uniform(0, 0.5)
                        await asyncio.sleep(jitter)
                        pose_stream_tasks[camera_id] = asyncio.create_task(
                            connect_to_pose_stream(camera_id)
                        )
            
            # 요청 간격 조절
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # 정상적인 취소, 조용히 처리
        pass
    except Exception:
        # 다른 예외, 조용히 처리
        pass
    finally:
        # 모든 스트림 태스크 취소
        for task in list(stream_tasks.values()) + list(pose_stream_tasks.values()):
            if not task.done():
                task.cancel()
        
        # 사용했던 세션 정리
        await utils.close_session()

# WebSocket 연결 및 이미지 스트리밍 수신 - 안정성 개선
async def connect_to_camera_stream(camera_id):
    """WebSocket을 통해 카메라 스트림에 연결"""
    # 연결 상태 업데이트
    update_connection_status(camera_id, "reconnecting")
    
    # 최대 재연결 시도 횟수 확인
    if (camera_id in app_state.connection_attempts and 
            app_state.connection_attempts[camera_id] >= config.MAX_RECONNECT_ATTEMPTS):
        # 지수 백오프 지연 계산
        delay = min(30, config.RECONNECT_DELAY * (2 ** app_state.connection_attempts[camera_id]))
        await asyncio.sleep(delay)
    
    # 재연결 시도 횟수 증가
    if camera_id not in app_state.connection_attempts:
        app_state.connection_attempts[camera_id] = 0
    app_state.connection_attempts[camera_id] += 1
    
    try:
        session = await utils.init_session()
        # WebSocket URL 구성
        ws_url = f"{config.API_URL.replace('http://', 'ws://')}/ws/stream/{camera_id}"
        
        # 향상된 WebSocket 옵션
        heartbeat = 30.0  # 30초 핑/퐁
        ws_timeout = aiohttp.ClientWSTimeout(ws_close=60.0)  # WebSocket 종료 대기 시간 60초
        
        async with session.ws_connect(
            ws_url, 
            heartbeat=heartbeat,
            timeout=ws_timeout,
            max_msg_size=0,  # 무제한
            compress=False  # 웹소켓 압축 비활성화로 성능 향상
        ) as ws:
            # 연결 성공 - 상태 업데이트 및 시도 횟수 초기화
            update_connection_status(camera_id, "connected")
            app_state.connection_attempts[camera_id] = 0
            
            last_ping_time = time.time()
            ping_interval = 25  # 25초마다 핑 전송 (30초 하트비트보다 짧게)
            
            while app_state.is_running:
                # 핑 전송 (주기적으로) - 서버 핑/퐁 메커니즘과 별개로 유지
                current_time = time.time()
                if current_time - last_ping_time >= ping_interval:
                    try:
                        await ws.ping()
                        last_ping_time = current_time
                    except:
                        # 핑 실패 시 루프 탈출하여 재연결
                        break
                
                # 데이터 수신 (짧은 타임아웃으로 반응성 유지)
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
                    
                    if msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        # 핑/퐁 처리
                        if msg.data == "ping":
                            await ws.send_str("pong")
                            continue
                        elif msg.data == "pong":
                            continue
                        
                        # JSON 메시지 처리
                        try:
                            data = json.loads(msg.data)
                            if data.get("type") == "image":
                                # 이미지 데이터 처리
                                image_data = data.get("image_data")
                                if image_data:
                                    # 이미지 디코딩 및 처리
                                    loop = utils.get_event_loop()
                                    future = loop.run_in_executor(
                                        utils.thread_pool, 
                                        utils.process_image_in_thread, 
                                        image_data
                                    )
                                    image = await future
                                    
                                    if image is not None:
                                        # 타임스탬프 파싱
                                        try:
                                            timestamp = datetime.fromisoformat(data.get("timestamp"))
                                        except (ValueError, TypeError):
                                            timestamp = datetime.now()
                                        
                                        # 동기화 버퍼에 저장
                                        frame_data = {
                                            "image": image,
                                            "time": timestamp
                                        }
                                        
                                        if camera_id in app_state.sync_buffer:
                                            app_state.sync_buffer[camera_id].append(frame_data)
                                        
                                        # 이미지 큐에도 저장
                                        if camera_id not in app_state.image_queues:
                                            app_state.image_queues[camera_id] = queue.Queue(maxsize=1)
                                        
                                        if not app_state.image_queues[camera_id].full():
                                            app_state.image_queues[camera_id].put(frame_data)
                        except json.JSONDecodeError:
                            # JSON 오류 무시
                            pass
                except asyncio.TimeoutError:
                    # 타임아웃은 정상이므로 무시
                    pass
    except asyncio.TimeoutError:
        # 연결 타임아웃
        update_connection_status(camera_id, "disconnected")
    except aiohttp.ClientConnectorError:
        # 서버 연결 실패
        update_connection_status(camera_id, "disconnected")
    except Exception:
        # 기타 예외
        update_connection_status(camera_id, "disconnected")
    
    # 함수 종료시 연결 해제 상태로 설정
    update_connection_status(camera_id, "disconnected")
    
    # 지수 백오프로 재연결 지연 계산 (최대 30초)
    backoff_delay = min(30, config.RECONNECT_DELAY * (2 ** (app_state.connection_attempts[camera_id] - 1)))
    await asyncio.sleep(backoff_delay)
    
    return False

# 포즈 데이터 WebSocket 연결 및 수신
async def connect_to_pose_stream(camera_id):
    """WebSocket을 통해 카메라의 포즈 데이터 스트림에 연결"""
    # 연결 상태 업데이트
    update_connection_status(camera_id, "reconnecting")
    
    try:
        session = await utils.init_session()
        # WebSocket URL 구성
        ws_url = f"{config.API_URL.replace('http://', 'ws://')}/ws/pose/{camera_id}"
        
        # 향상된 WebSocket 옵션
        heartbeat = 30.0  # 30초 핑/퐁
        ws_timeout = aiohttp.ClientWSTimeout(ws_close=60.0)
        
        async with session.ws_connect(
            ws_url, 
            heartbeat=heartbeat,
            timeout=ws_timeout,
            max_msg_size=0,
            compress=False
        ) as ws:
            # 연결 성공 - 상태 업데이트
            update_connection_status(camera_id, "connected")
            app_state.connection_attempts[camera_id] = 0
            
            last_ping_time = time.time()
            ping_interval = 25  # 25초마다 핑 전송
            
            while app_state.is_running:
                # 핑 전송 (주기적으로)
                current_time = time.time()
                if current_time - last_ping_time >= ping_interval:
                    try:
                        await ws.ping()
                        last_ping_time = current_time
                    except:
                        # 핑 실패 시 루프 탈출하여 재연결
                        break
                
                # 데이터 수신
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
                    
                    if msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        # 핑/퐁 처리
                        if msg.data == "ping":
                            await ws.send_str("pong")
                            continue
                        elif msg.data == "pong":
                            continue
                        
                        # JSON 메시지 처리
                        try:
                            data = json.loads(msg.data)
                            if data.get("type") == "pose_data":
                                # 포즈 데이터 처리
                                pose_data = data.get("pose_data")
                                timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
                                
                                if pose_data:
                                    # 포즈 데이터 저장
                                    app_state.pose_data_store[camera_id] = pose_data
                                    app_state.pose_update_times[camera_id] = timestamp
                        except json.JSONDecodeError:
                            # JSON 오류 무시
                            pass
                except asyncio.TimeoutError:
                    # 타임아웃은 정상이므로 무시
                    pass
    except Exception:
        # 연결 오류
        update_connection_status(camera_id, "disconnected")
    
    # 함수 종료시 연결 해제 상태로 설정
    update_connection_status(camera_id, "disconnected")
    
    return False

# 스레드 시작 함수
def start_websocket_thread():
    """WebSocket 처리 스레드 시작"""
    if not app_state.is_running:
        app_state.is_running = True
    
    # 스레드 시작
    thread = threading.Thread(target=run_async_loop, daemon=True)
    thread.start()
    return thread 