import streamlit as st
import asyncio
import websockets
import json
import queue
import threading
from datetime import datetime
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh

# session_state에 ws_queue가 없다면 생성하여 저장
if "ws_queue" not in st.session_state:
    st.session_state.ws_queue = queue.Queue()

# ws_queue를 지역 변수로 할당 (세션 상태 내 객체를 참조)
ws_queue = st.session_state.ws_queue

# 수신된 메시지 로그를 저장 (이미 session_state에 저장됨)
if "ws_messages" not in st.session_state:
    st.session_state.ws_messages = []

def start_ws_thread():
    """백그라운드에서 WebSocket 연결을 유지하는 스레드 시작"""
    def run_ws():
        asyncio.run(ws_handler())
    thread = threading.Thread(target=run_ws, daemon=True)
    thread.start()

async def ws_handler():
    """WebSocket 연결을 맺고 서버 메시지를 수신"""
    uri = "ws://localhost:8000/ws/updates"  # 서버 WebSocket 엔드포인트
    try:
        async with websockets.connect(uri) as websocket:
            print("DEBUG: WebSocket 연결 성공")
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                print("DEBUG: WebSocket 메시지 수신:", data)
                ws_queue.put(data)
                print("DEBUG: WS 메시지 큐에 추가됨:", data)
    except Exception as e:
        print(f"DEBUG: WebSocket 연결 오류: {e}")

def main():
    st.set_page_config(page_title="KOMI 웹캠 모니터링 (WebSocket)", layout="wide")
    st.title("KOMI 웹캠 모니터링 (WebSocket)")

    # 백그라운드 WebSocket 스레드 시작 (한 번만 시작)
    if "ws_thread_started" not in st.session_state:
        st.session_state.ws_thread_started = True
        start_ws_thread()

    # 자동 새로고침 (1초마다)
    st_autorefresh(interval=1000, key="ws_autorefresh")

    # 큐에 쌓인 메시지를 처리하여 session_state.ws_messages에 추가
    while not ws_queue.empty():
        msg = ws_queue.get()
        print("DEBUG: 메인 스레드에서 큐 메시지 처리:", msg)
        st.session_state.ws_messages.append(msg)

    st.subheader("수신된 WebSocket 메시지 로그")
    st.write(st.session_state.ws_messages)

if __name__ == "__main__":
    main()
