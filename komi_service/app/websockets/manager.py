"""
웹소켓 연결 관리자
"""

import asyncio
from fastapi import WebSocket
from typing import Set, List, Dict, Any, Optional
from datetime import datetime

from app.utils.websocket_utils import broadcast_message, create_message

class WebSocketManager:
    """웹소켓 연결을 관리하는 클래스"""
    
    def __init__(self):
        # 일반 웹소켓 연결 관리
        self.active_connections: Set[WebSocket] = set()
        
        # 정리 작업 주기
        self.cleanup_interval = 60  # 초
        self.last_cleanup = datetime.now()
    
    async def connect(self, websocket: WebSocket) -> None:
        """웹소켓 연결 처리
        
        Args:
            websocket: 웹소켓 연결
        """
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket) -> None:
        """웹소켓 연결 해제
        
        Args:
            websocket: 웹소켓 연결
        """
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """모든 연결에 메시지 브로드캐스트
        
        Args:
            message: 전송할 메시지
        """
        dead_connections = await broadcast_message(self.active_connections, message)
        
        # 연결 끊긴 웹소켓 정리
        for websocket in dead_connections:
            self.disconnect(websocket)
    
    async def cleanup_dead_connections(self) -> None:
        """연결 끊긴 웹소켓 정리"""
        dead_connections = set()
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text("ping")
            except Exception:
                dead_connections.add(websocket)
        
        # 연결 끊긴 웹소켓 정리
        for websocket in dead_connections:
            self.disconnect(websocket)
    
    async def periodic_cleanup(self) -> None:
        """주기적인 정리 작업"""
        while True:
            now = datetime.now()
            if (now - self.last_cleanup).total_seconds() >= self.cleanup_interval:
                await self.cleanup_dead_connections()
                self.last_cleanup = now
            
            await asyncio.sleep(10)
    
    def get_connection_count(self) -> int:
        """활성 연결 수 반환"""
        return len(self.active_connections)
    
    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]) -> bool:
        """JSON 데이터 전송
        
        Args:
            websocket: 웹소켓 연결
            data: 전송할 JSON 데이터
            
        Returns:
            전송 성공 여부
        """
        try:
            await websocket.send_json(data)
            return True
        except Exception:
            self.disconnect(websocket)
            return False
    
    async def send_text(self, websocket: WebSocket, text: str) -> bool:
        """텍스트 데이터 전송
        
        Args:
            websocket: 웹소켓 연결
            text: 전송할 텍스트
            
        Returns:
            전송 성공 여부
        """
        try:
            await websocket.send_text(text)
            return True
        except Exception:
            self.disconnect(websocket)
            return False

# 전역 웹소켓 관리자 인스턴스
websocket_manager = WebSocketManager() 