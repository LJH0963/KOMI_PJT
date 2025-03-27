"""
웹소켓 관련 유틸리티 함수
"""

import asyncio
import json
from datetime import datetime
from fastapi import WebSocket
from typing import Set, Dict, Optional, Any

# 활성 웹소켓 연결 관리
active_connections: Set[WebSocket] = set()

async def keep_websocket_alive(websocket: WebSocket, ping_interval: int = 30, max_idle_time: int = 60) -> bool:
    """WebSocket 연결을 유지하는 함수
    
    Args:
        websocket: 웹소켓 연결
        ping_interval: 핑 전송 간격 (초)
        max_idle_time: 최대 허용 대기 시간 (초)
        
    Returns:
        연결 유지 성공 여부
    """
    last_ping_time = asyncio.get_event_loop().time()
    last_received_time = asyncio.get_event_loop().time()
    
    try:
        while True:
            current_time = asyncio.get_event_loop().time()
            
            # 마지막 응답으로부터 너무 오래 경과했는지 확인
            if current_time - last_received_time > max_idle_time:
                # 연결이 너무 오래 idle 상태임
                return False
            
            # 정기적인 핑 전송
            if current_time - last_ping_time >= ping_interval:
                try:
                    # 핑 메시지 전송
                    await websocket.send_text("ping")
                    last_ping_time = current_time
                except Exception:
                    # 핑 전송 실패
                    return False
            
            # 메시지 수신 시도 (짧은 타임아웃으로 반응성 유지)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                last_received_time = asyncio.get_event_loop().time()  # 메시지 수신 시간 업데이트
                
                # 핑/퐁 처리
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "pong":
                    # 클라이언트에서 보낸 퐁 응답
                    pass
            except asyncio.TimeoutError:
                # 타임아웃은 정상적인 상황, 계속 진행
                pass
            except Exception:
                # 기타 오류 발생 시 연결 종료
                return False
            
            # 잠시 대기 후 다음 루프
            await asyncio.sleep(0.1)
    except Exception:
        return False
    
    return True

async def broadcast_message(connections: Set[WebSocket], message: Dict[str, Any]) -> Set[WebSocket]:
    """웹소켓 연결 목록에 메시지를 브로드캐스트
    
    Args:
        connections: 웹소켓 연결 목록
        message: 전송할 메시지 (JSON 직렬화 가능한 사전)
        
    Returns:
        연결 끊긴 웹소켓 목록
    """
    message_str = json.dumps(message)
    dead_connections = set()
    
    for websocket in connections:
        try:
            await websocket.send_text(message_str)
        except Exception:
            dead_connections.add(websocket)
    
    return dead_connections

async def broadcast_json(connections: Set[WebSocket], message: Dict[str, Any]) -> Set[WebSocket]:
    """웹소켓 연결 목록에 JSON 메시지를 브로드캐스트
    
    Args:
        connections: 웹소켓 연결 목록
        message: 전송할 JSON 메시지
        
    Returns:
        연결 끊긴 웹소켓 목록
    """
    dead_connections = set()
    
    for websocket in connections:
        try:
            await websocket.send_json(message)
        except Exception:
            dead_connections.add(websocket)
    
    return dead_connections

def create_message(type_: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """웹소켓 메시지 생성 헬퍼 함수
    
    Args:
        type_: 메시지 타입
        data: 메시지 데이터
        
    Returns:
        완성된 메시지 사전
    """
    message = {
        "type": type_,
        "timestamp": datetime.now().isoformat(),
        **data
    }
    return message 