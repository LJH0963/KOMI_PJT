import socket

# 서버 정보
HOST = '127.0.0.1'
PORT = 12345

# 소켓 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 서버에 연결
client_socket.connect((HOST, PORT))
print("서버에 연결되었습니다. 종료하려면 'exit'를 입력하세요.")

count = 0
try:
    while True:
        # 사용자 입력 받기
        message = f'--------------------------{count}'
        count += 1        
        # 'exit' 입력 시 종료
        if message.lower() == 'exit':
            print("클라이언트를 종료합니다.")
            break
        
        # 서버로 메시지 전송
        client_socket.send(message.encode())

        # 서버 응답 수신
        response = client_socket.recv(1024).decode()
        print(f"서버 응답: {response}")

except Exception as e:
    print(f"오류 발생: {e}")

finally:
    # 소켓 종료
    client_socket.close()

