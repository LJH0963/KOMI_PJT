import socket

# 서버 정보
HOST = '127.0.0.1'  # 연결할 서버의 IP 주소 (로컬호스트)
PORT = 12345       # 연결할 서버의 포트 번호 (서버와 동일한 포트 사용)

# 소켓 생성 (TCP 클라이언트 소켓)
# AF_INET: IPv4 주소 체계 사용
# SOCK_STREAM: TCP 프로토콜 사용 (연결 지향적 통신)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 서버에 연결 시도
# connect() 함수는 지정된 IP와 포트로 서버에 연결을 요청
client_socket.connect((HOST, PORT))
print("서버에 연결되었습니다. 종료하려면 'exit'를 입력하세요.")

count = 0  # 메시지 카운터 (자동 메시지 생성에 사용)
try:
    while True:
        # 자동으로 메시지 생성 (카운터를 사용한 일련번호 포함)
        message = f'--------------------------{count}'
        count += 1        
        
        # 'exit' 입력 시 루프 종료 및 프로그램 종료
        if message.lower() == 'exit':
            print("클라이언트를 종료합니다.")
            break
        
        # 서버로 메시지 전송
        # send() 함수로 메시지를 바이트 형식으로 인코딩하여 전송
        client_socket.send(message.encode())

        # 서버 응답 수신 (최대 1024바이트)
        # recv() 함수로 서버의 응답을 받아 디코딩하여 문자열로 변환
        response = client_socket.recv(1024).decode()
        print(f"서버 응답: {response}")

except Exception as e:
    # 예외 발생 시 오류 메시지 출력
    print(f"오류 발생: {e}")

finally:
    # 프로그램 종료 시 소켓 종료 (자원 해제)
    client_socket.close()

