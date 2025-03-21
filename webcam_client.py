import aiohttp
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class AsyncWebcamClient:
    def __init__(self, server_url: str = "http://localhost:8000", 
                 websocket_url: str = "ws://localhost:8000/ws/pose_stream",
                 timeout: float = 30.0):  # 타임아웃 값 증가 (기본 30초)
        self.server_url = server_url
        self.websocket_url = websocket_url
        self.timeout = timeout
        self.cap = None
        self.ws = None
        self.running = False
    
    async def connect(self):
        """서버에 WebSocket 연결 설정"""
        # timeout 매개변수를 추가하여 충분한 시간 제공
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        self.ws = await self.session.ws_connect(self.websocket_url)
        return self.ws is not None
    
    # HTTP 요청을 위한 별도 메서드 (WebSocket이 아닌 일반 HTTP 요청에 사용)
    async def send_http_request(self, endpoint: str, data: dict):
        """HTTP 요청을 보내는 비동기 메서드"""
        url = f"{self.server_url}/{endpoint}"
        try:
            async with self.session.post(url, json=data, timeout=self.timeout) as response:
                return await response.json()
        except asyncio.TimeoutError:
            print(f"HTTP 요청 타임아웃 (endpoint: {endpoint})")
            return {"status": "error", "message": "Request timeout"}

    async def run_stream(self):
        """지속적인 프레임 캡처 및 전송 실행"""
        self.running = True
        try:
            while self.running:
                frame_data = await self.process_frame()
                if frame_data:
                    try:
                        # WebSocket 요청 시도
                        response = await self.send_frame_websocket(frame_data)
                        
                        # 응답 처리
                        if hasattr(self, 'on_response') and callable(self.on_response):
                            self.on_response(response)
                        
                        # 실시간 처리를 위한 적절한 딜레이 조정
                        await asyncio.sleep(0.05)  # 필요에 따라 조정 (높은 값 = 낮은 FPS)
                    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                        print(f"스트리밍 통신 오류: {str(e)}, 재연결 시도 중...")
                        # 연결 끊김 시 재연결 시도
                        await self.reconnect()
        except Exception as e:
            print(f"스트리밍 오류: {str(e)}")
        finally:
            await self.close()
    
    async def reconnect(self):
        """연결 끊김 시 재연결 시도"""
        try:
            # 기존 연결 정리
            if self.ws:
                await self.ws.close()
            
            # 세션 재생성
            if hasattr(self, 'session') and not self.session.closed:
                await self.session.close()
            
            # 새로운 세션과 연결 설정
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            self.ws = await self.session.ws_connect(self.websocket_url)
            print("웹소켓 재연결 성공")
        except Exception as e:
            print(f"재연결 실패: {str(e)}")
            # 잠시 대기 후 다음 시도
            await asyncio.sleep(1)

    async def close(self):
        """WebSocket 연결 및 세션 정리"""
        if self.ws:
            await self.ws.close()
        if hasattr(self, 'session') and not self.session.closed:
            await self.session.close()
        self.running = False
        self.cap = None
        self.ws = None
        print("웹캠 클라이언트 정리 완료") 