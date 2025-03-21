import socket

# 서버 정보
HOST = '127.0.0.1'  # 로컬호스트 (자기 자신의 IP 주소)
PORT = 12345        # 포트 번호 (임의 선택한 번호)

# UDP 소켓 생성
# AF_INET: IPv4 주소 체계 사용
# SOCK_DGRAM: UDP 프로토콜 사용 (연결 없는 통신 방식)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))  # 소켓을 지정된 주소와 포트에 바인딩

print(f"UDP 서버가 {HOST}:{PORT}에서 실행 중입니다...")

clients = set()  # 클라이언트 주소를 저장할 집합(중복 없음)

while True:  # 웹캠의 경우
    try:
        # 클라이언트로부터 데이터 수신 (최대 1024바이트)
        # UDP는 비연결 지향적이므로 recvfrom 함수를 사용하여 데이터와 발신자 주소를 함께 수신
        data, client_addr = server_socket.recvfrom(1024)

        # 새로운 클라이언트인 경우 목록에 추가
        if client_addr not in clients:
            clients.add(client_addr)
            print(f"새 클라이언트 {client_addr} 연결됨.")

        # 받은 데이터 디코딩 (바이트 -> 문자열)
        message = data.decode()
        print(f"클라이언트({client_addr})로부터 받은 데이터: {message}")

        # 모든 클라이언트에게 메시지 브로드캐스트 (모든 클라이언트에게 전송)
        # UDP에서는 각 클라이언트에게 개별적으로 sendto를 사용하여 데이터 전송
        response = f"[서버 응답] {client_addr}의 메시지: {message}"
        for client in clients:
            server_socket.sendto(response.encode(), client)  # 문자열을 바이트로 인코딩하여 전송
            print(f"클라이언트 {client}에게 응답 전송")  # 중간에 동시에 시작 혹은 종료 명령을 할 때 필요

    except Exception as e:
        print(f"오류 발생: {e}")
        # 예외 발생시에도 서버 종료 없이 계속 실행
        
# UDP의 경우 스레드가 필요하지 않음