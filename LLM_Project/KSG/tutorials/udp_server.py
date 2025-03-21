import socket

# 서버 정보
HOST = '127.0.0.1'  # 로컬호스트
PORT = 12345        # 포트 번호

# UDP 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

print(f"UDP 서버가 {HOST}:{PORT}에서 실행 중입니다...")

clients = set()  # 클라이언트 목록 저장

while True:
    try:
        # 클라이언트로부터 데이터 수신 (최대 1024바이트)
        data, client_addr = server_socket.recvfrom(1024)

        # 클라이언트가 새로 연결되었으면 목록에 추가
        if client_addr not in clients:
            clients.add(client_addr)
            print(f"새 클라이언트 {client_addr} 연결됨.")

        # 받은 데이터 디코딩
        message = data.decode()
        print(f"클라이언트({client_addr})로부터 받은 데이터: {message}")

        # 모든 클라이언트에게 메시지 브로드캐스트 (선택사항)
        response = f"[서버 응답] {client_addr}의 메시지: {message}"
        for client in clients:
            server_socket.sendto(response.encode(), client)
            print(f"클라이언트 {client}에게 응답 전송")

    except Exception as e:
        print(f"오류 발생: {e}")