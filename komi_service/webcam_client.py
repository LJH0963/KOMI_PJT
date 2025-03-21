import cv2
import base64
import json
import time
import websocket
import sys
import threading
import argparse
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_SERVER_URL = "ws://localhost:8000/ws/camera"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FPS = 10  # 초당 프레임 수
DEFAULT_RESOLUTION = (160, 120)  # 해상도

class WebcamClient:
    """웹캠 이미지를 캡처하여 WebSocket으로 전송하는 클라이언트"""
    def __init__(self, server_url=DEFAULT_SERVER_URL, camera_index=DEFAULT_CAMERA_INDEX, fps=DEFAULT_FPS, resolution=DEFAULT_RESOLUTION):
        self.server_url = server_url
        self.camera_index = camera_index
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.resolution = resolution
        self.ws = None
        self.running = False
        self.cap = None
        self.connected = False
        self.client_id = None
        self.capture_thread = None
    
    def connect(self):
        """WebSocket 서버에 연결"""
        print(f"서버에 연결 중... ({self.server_url})")
        self.ws = websocket.WebSocketApp(
            self.server_url,
            on_open=lambda ws: self._on_open(ws),
            on_message=lambda ws, msg: self._on_message(ws, msg),
            on_error=lambda ws, error: self._on_error(ws, error),
            on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg)
        )
        
        # WebSocket 연결 시작 (별도 스레드)
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # 연결 대기
        for _ in range(50):  # 5초 타임아웃
            if self.connected:
                return True
            time.sleep(0.1)
        
        print("서버 연결 타임아웃")
        return False
    
    def _on_open(self, ws):
        """WebSocket 연결 성공 시 호출"""
        print("서버에 연결되었습니다.")
        self.connected = True
        self.client_id = None
        logger.info(f"카메라 ID: {self.client_id}")
    
    def _on_message(self, ws, message):
        """서버로부터 메시지 수신 시 호출"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "connection_successful":
                self.client_id = data.get("camera_id")
                logger.info(f"카메라 ID: {self.client_id}")
            elif msg_type == "frame_processed":
                pose_data = data.get("pose_data", {})
                if pose_data and "pose" in pose_data:
                    print(f"포즈 감지됨 - 키포인트: {len(pose_data['pose'][0].get('keypoints', []))}개")
        except Exception as e:
            print(f"메시지 처리 오류: {e}")
    
    def _on_error(self, ws, error):
        """웹소켓 오류 발생 시 호출"""
        logger.error(f"웹소켓 오류: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket 연결 종료 시 호출"""
        print("서버 연결이 종료되었습니다.")
        self.connected = False
    
    def find_available_cameras(self):
        """사용 가능한 카메라 목록 확인"""
        available_cameras = []
        for i in range(10):  # 0부터 9까지의 카메라 인덱스 확인
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"카메라 {i} - 사용 가능")
                cap.release()
            time.sleep(0.1)  # 카메라 초기화 시간
            
        return available_cameras
    
    def start_capture(self):
        """카메라 시작"""
        if self.running:
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"카메라 {self.camera_index}를 열 수 없습니다.")
                # 다른 카메라 인덱스 시도
                for i in range(3):
                    if i == self.camera_index:
                        continue
                    
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        self.camera_index = i
                        print(f"카메라 {i}에 성공적으로 연결했습니다.")
                        return True
                return False
            
            # 해상도 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            print(f"카메라 {self.camera_index} 시작됨 (FPS: {self.fps})")
            
            # 상태 설정 및 캡처 스레드 시작
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            return True
        except Exception as e:
            print(f"카메라 시작 오류: {e}")
            return False
    
    def _capture_loop(self):
        """카메라 캡처 루프"""
        last_capture_time = 0
        
        while self.running and self.connected:
            try:
                # 프레임 간격 조절
                current_time = time.time()
                if current_time - last_capture_time < self.frame_interval:
                    time.sleep(0.01)  # CPU 사용량 감소
                    continue
                
                # 프레임 캡처
                ret, frame = self.cap.read()
                if not ret:
                    print("프레임을 캡처할 수 없습니다.")
                    time.sleep(0.1)
                    continue
                
                # 프레임 인코딩 및 전송
                self._send_frame(frame)
                
                # 마지막 캡처 시간 업데이트
                last_capture_time = time.time()
            except Exception as e:
                print(f"캡처 오류: {e}")
                time.sleep(0.1)
    
    def _send_frame(self, frame):
        """프레임을 서버로 전송"""
        if not self.connected:
            return
        
        try:
            # JPEG으로 인코딩
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Base64로 변환
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            # 웹소켓으로 전송
            message = {
                "type": "frame",
                "image_data": img_base64,
                "timestamp": time.time()
            }
            
            self.ws.send(json.dumps(message))
        except Exception as e:
            print(f"프레임 전송 오류: {e}")
    
    def run(self):
        """클라이언트 실행"""
        print("===== 웹캠 클라이언트 시작 =====")
        
        # 서버 연결
        if not self.connect():
            return
        
        # 카메라 시작
        if not self.start_capture():
            self.stop()
            return
        
        # 메인 루프 시작
        self.running = True
        try:
            print("웹캠 스트리밍 시작. 중지하려면 Ctrl+C를 누르세요.")
            
            last_time = time.time()
            frames_sent = 0
            
            while self.running:
                current_time = time.time()
                if current_time - last_time >= self.frame_interval:
                    if self._send_frame():
                        frames_sent += 1
                        # 매 100프레임마다 FPS 계산
                        if frames_sent % 100 == 0:
                            elapsed = current_time - last_time
                            fps = 100 / elapsed if elapsed > 0 else 0
                            print(f"전송 속도: {fps:.1f} FPS")
                            last_time = current_time
                            frames_sent = 0
                        else:
                            last_time = current_time
                    
                # CPU 사용량 줄이기
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다.")
        finally:
            self.stop()
    
    def stop(self):
        """클라이언트 종료"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(2)
        if self.cap:
            self.cap.release()
        if self.ws:
            self.ws.close()
        print("웹캠 클라이언트가 종료되었습니다.")

def select_camera():
    """사용 가능한 카메라 목록에서 선택"""
    print("사용 가능한 카메라를 확인 중...")
    
    # 임시 클라이언트 생성
    client = WebcamClient()
    available_cameras = client.find_available_cameras()
    
    if not available_cameras:
        print("사용 가능한 카메라가 없습니다.")
        return 0
    
    if len(available_cameras) == 1:
        print(f"카메라 {available_cameras[0]}만 사용 가능합니다. 자동 선택됨.")
        return available_cameras[0]
    
    # 카메라 선택 UI
    print("\n사용할 카메라를 선택하세요:")
    for idx, cam_idx in enumerate(available_cameras):
        print(f"{idx+1}. 카메라 {cam_idx}")
    
    choice = input("번호 입력 (기본값: 1): ").strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(available_cameras):
            return available_cameras[idx]
    except (ValueError, IndexError):
        pass
    
    # 기본값
    return available_cameras[0]

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='KOMI 웹캠 클라이언트')
    parser.add_argument('--server', type=str, default=DEFAULT_SERVER_URL,
                        help=f'웹소켓 서버 URL (기본값: {DEFAULT_SERVER_URL})')
    parser.add_argument('--camera', type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f'카메라 인덱스 (기본값: {DEFAULT_CAMERA_INDEX})')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                        help=f'초당 프레임 수 (기본값: {DEFAULT_FPS})')
    parser.add_argument('--width', type=int, default=DEFAULT_RESOLUTION[0],
                        help=f'이미지 너비 (기본값: {DEFAULT_RESOLUTION[0]})')
    parser.add_argument('--height', type=int, default=DEFAULT_RESOLUTION[1],
                        help=f'이미지 높이 (기본값: {DEFAULT_RESOLUTION[1]})')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 클라이언트 생성
    client = WebcamClient(
        server_url=args.server,
        camera_index=args.camera,
        fps=args.fps,
        resolution=(args.width, args.height)
    )
    
    try:
        logger.info(f"서버 {args.server}에 연결 중...")
        
        # 서버 연결
        if not client.connect():
            logger.error("서버에 연결할 수 없습니다.")
            sys.exit(1)
        
        logger.info(f"카메라 {args.camera} 캡처 시작 중...")
        
        # 카메라 캡처 시작
        if not client.start_capture():
            logger.error("카메라 캡처를 시작할 수 없습니다.")
            client.stop()
            sys.exit(1)
        
        logger.info(f"웹캠 스트리밍 중... (Ctrl+C로 종료)")
        
        # 메인 스레드는 종료되지 않도록 대기
        while client.connected and client.running:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("사용자에 의해 종료되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
    finally:
        # 종료 전 정리
        client.stop()
        logger.info("클라이언트가 종료되었습니다.")

if __name__ == "__main__":
    main() 