import socket
import threading

# 서버 정보 설정
HOST = '127.0.0.1'  # 로컬호스트 IP 주소 (서버 자신의 IP)
PORT = 12345       # 사용할 포트 번호

# 소켓 생성
# AF_INET: IPv4 주소 체계 사용
# SOCK_STREAM: TCP 프로토콜 사용 (연결 지향적 통신)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP/IP
server_socket.bind((HOST, PORT))  # 소켓을 지정된 주소와 포트에 바인딩
server_socket.listen(5)           # 최대 5개의 연결 요청을 대기열에 유지

print(f"서버가 {HOST}:{PORT}에서 대기 중입니다...")

# 클라이언트 처리 함수 (각 클라이언트마다 별도의 스레드로 실행됨)
def handle_client(client_socket, addr):
    print(f"클라이언트 {addr} 연결됨.")
    try:
        while True:
            # 클라이언트로부터 데이터 수신 (최대 1024바이트)
            # TCP는 연결 지향적이므로 recv 함수만 사용 (발신자 주소는 이미 알고 있음)
            data = client_socket.recv(1024).decode()
            
            # 클라이언트가 'exit' 입력하거나 연결 종료 시 루프 탈출
            if not data or data.lower() == 'exit':
                print(f"클라이언트 {addr} 연결 종료.")
                break
            
            print(f"클라이언트({addr})로부터 받은 데이터: {data}")

            # 클라이언트에게 응답 메시지 전송
            # TCP에서는 이미 연결된 소켓을 통해 send 함수로 직접 데이터 전송
            response = f"서버에서 받은 데이터: {data}"
            client_socket.send(response.encode())  # 문자열을 바이트로 인코딩하여 전송

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # 클라이언트 소켓 종료 (자원 해제)
        client_socket.close()

# 메인 서버 루프: 클라이언트 연결 수락 및 처리
while True:
    # accept()는 새 클라이언트 연결을 수락하고 새 소켓과 주소를 반환
    # 이 함수는 새 연결이 들어올 때까지 블로킹됨
    client_socket, addr = server_socket.accept()
    
    # 각 클라이언트를 개별 스레드로 처리 (동시 다중 클라이언트 처리)
    client_thread = threading.Thread(target=handle_client, args=(client_socket, addr))
    client_thread.daemon = True  # 메인 스레드 종료 시 함께 종료되도록 데몬 스레드로 설정
    client_thread.start()        # 스레드 시작


# thread 필수
