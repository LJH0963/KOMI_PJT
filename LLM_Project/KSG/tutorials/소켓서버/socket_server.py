import socket
import threading

# 서버 정보 설정
HOST = '127.0.0.1'
PORT = 12345

# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print(f"서버가 {HOST}:{PORT}에서 대기 중입니다...")

# 클라이언트 처리 함수 (각 클라이언트마다 실행)
def handle_client(client_socket, addr):
    print(f"클라이언트 {addr} 연결됨.")
    try:
        while True:
            # 클라이언트로부터 데이터 수신
            data = client_socket.recv(1024).decode()
            
            # 클라이언트가 'exit' 입력 시 연결 종료
            if not data or data.lower() == 'exit':
                print(f"클라이언트 {addr} 연결 종료.")
                break
            
            print(f"클라이언트({addr})로부터 받은 데이터: {data}")

            # 클라이언트에게 응답 전송
            response = f"서버에서 받은 데이터: {data}"
            client_socket.send(response.encode())

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # 클라이언트 소켓 종료
        client_socket.close()

# 클라이언트 접속을 처리하는 루프
while True:
    client_socket, addr = server_socket.accept()
    
    # 각 클라이언트를 개별 스레드로 처리
    client_thread = threading.Thread(target=handle_client, args=(client_socket, addr))
    client_thread.start()
