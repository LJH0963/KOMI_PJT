import json
from typing import List, Dict, Any, Set
from fastapi import WebSocket

class WebSocketManager:
    """
    WebSocket 연결을 관리하는 클래스
    """
    def __init__(self):
        # 활성 클라이언트 목록
        self.active_connections: Set[WebSocket] = set()
        print("웹소켓 매니저 초기화됨")
    
    async def connect(self, websocket: WebSocket):
        """
        클라이언트 연결 수락
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"새 클라이언트 연결됨 (총 {len(self.active_connections)}개 연결)")
    
    def disconnect(self, websocket: WebSocket):
        """
        클라이언트 연결 해제
        """
        self.active_connections.discard(websocket)
        print(f"클라이언트 연결 해제됨 (총 {len(self.active_connections)}개 연결)")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """
        특정 클라이언트에 메시지 전송
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"개인 메시지 전송 오류: {str(e)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        모든 클라이언트에 메시지 브로드캐스트
        """
        disconnected_clients = set()
        
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"브로드캐스트 오류 (연결 해제됨): {str(e)}")
                disconnected_clients.add(websocket)
        
        # 연결이 끊어진 클라이언트 제거
        for websocket in disconnected_clients:
            self.disconnect(websocket)
    
    async def broadcast_pose_data(self, pose_data: Dict[str, Any], accuracy: float = 0, 
                             similarity_details: Dict[str, float] = None):
        """
        포즈 데이터 브로드캐스트
        """
        message = {
            "pose_data": pose_data,
            "accuracy": accuracy,
            "similarity_details": similarity_details or {}
        }
        
        await self.broadcast(message)
    
    async def send_exercise_list(self, exercises: List[Dict[str, Any]]):
        """
        운동 목록 브로드캐스트
        """
        message = {
            "type": "exercise_list",
            "exercises": exercises
        }
        await self.broadcast(message)
    
    @property
    def connected_clients_count(self) -> int:
        """
        연결된 클라이언트 수 반환
        """
        return len(self.active_connections)

# 전역 웹소켓 매니저 인스턴스
ws_manager = WebSocketManager()
