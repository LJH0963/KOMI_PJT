from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict

class WebSocketManager:
    """
    📌 웹소켓 연결을 관리하는 클래스
    - 여러 클라이언트 지원 (다중 연결)
    - 실시간 데이터 전송
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, message: Dict):
        """모든 활성 클라이언트에게 JSON 데이터 전송"""
        for connection in self.active_connections:
            await connection.send_json(message)

# 전역 웹소켓 매니저 인스턴스
ws_manager = WebSocketManager()
