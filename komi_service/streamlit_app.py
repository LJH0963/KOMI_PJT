import streamlit as st
import requests
import json
import time
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import io
import websocket
import threading
from datetime import datetime

# 서비스 URL 설정 (FastAPI 백엔드)
API_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001/pose/ws"

# 세션 상태 초기화
if 'exercise_history' not in st.session_state:
    st.session_state.exercise_history = []
if 'selected_exercise' not in st.session_state:
    st.session_state.selected_exercise = None
if 'accuracy_history' not in st.session_state:
    st.session_state.accuracy_history = []
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False
    st.session_state.ws_data = {
        "latest_pose": None,
        "accuracy": 0,
        "similarity_details": {}
    }

# 웹소켓 콜백 함수
def on_message(ws, message):
    """웹소켓 메시지 수신 시 호출되는 콜백"""
    try:
        data = json.loads(message)
        st.session_state.latest_message = data
        
        # 메시지 유형에 따른 처리
        if 'type' in data and data['type'] == 'exercise_list':
            st.session_state.exercise_list = data['exercises']
        elif 'pose_data' in data:
            st.session_state.ws_data['latest_pose'] = data['pose_data']
            if 'accuracy' in data:
                st.session_state.ws_data['accuracy'] = data['accuracy']
                # 정확도 기록 (최대 30개 항목)
                if len(st.session_state.accuracy_history) >= 30:
                    st.session_state.accuracy_history.pop(0)
                st.session_state.accuracy_history.append(data['accuracy'])
            if 'similarity_details' in data:
                st.session_state.ws_data['similarity_details'] = data['similarity_details']
    except Exception as e:
        st.error(f"메시지 처리 오류: {str(e)}")

def on_error(ws, error):
    """웹소켓 에러 발생 시 콜백"""
    st.error(f"웹소켓 오류: {str(error)}")

def on_close(ws, close_status_code, close_reason):
    """웹소켓 연결 종료 시 콜백"""
    st.session_state.ws_connected = False

def on_open(ws):
    """웹소켓 연결 성공 시 콜백"""
    st.session_state.ws_connected = True

# 웹소켓 연결 함수
def connect_websocket():
    """웹소켓 연결 함수"""
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # 별도 스레드에서 웹소켓 실행
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    return ws

# 더미 이미지 생성 함수
def generate_dummy_image(size=(640, 480)):
    """더미 이미지 생성"""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # 간단한 원 그리기
    cv2.circle(img, (size[0]//2, size[1]//2), 50, (0, 255, 0), -1)
    return img

# 이미지 전송 함수
def send_image(exercise_type=None):
    """더미 이미지를 서버로 전송"""
    # 더미 이미지 생성
    img = generate_dummy_image()
    
    # 이미지를 JPEG로 변환
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    
    # API 요청 데이터 구성
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    data = {}
    if exercise_type:
        data["exercise_type"] = exercise_type
    
    try:
        # 이미지 업로드 API 호출
        response = requests.post(f"{API_URL}/pose/upload", files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API 오류: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"이미지 전송 오류: {str(e)}")
        return None

# 메인 UI
def main():
    """메인 Streamlit UI"""
    st.set_page_config(page_title="KOMI - AI 자세 분석", layout="wide")
    
    # 웹소켓 연결
    if not st.session_state.ws_connected:
        ws = connect_websocket()
    
    # 페이지 헤더
    st.title("🧘 KOMI - AI 자세 분석")
    st.markdown("### 인공지능을 활용한 실시간 자세 분석 및 피드백 시스템")
    
    # 메인 컨텐츠 영역
    col1, col2 = st.columns([2, 1])
    
    # 왼쪽 컬럼 - 웹캠 영상 및 포즈 분석
    with col1:
        st.subheader("📹 실시간 자세 분석")
        
        # 카메라 활성화 버튼
        camera_on = st.toggle("카메라 활성화", True)
        
        # 카메라 피드 영역
        cam_placeholder = st.empty()
        
        # 정확도 표시 영역
        accuracy_gauge = st.empty()
        
        # 그래프 영역
        chart_area = st.empty()
    
    # 오른쪽 컬럼 - 운동 선택 및 가이드
    with col2:
        st.subheader("🏋️ 운동 선택")
        
        # 운동 유형 목록
        exercises = [
            {"id": "shoulder", "name": "어깨 운동", "description": "어깨 통증 완화 운동"},
            {"id": "knee", "name": "무릎 운동", "description": "무릎 관절 강화 운동"},
            {"id": "posture", "name": "자세 교정", "description": "바른 자세 교정 운동"}
        ]
        
        # 운동 선택 버튼
        for ex in exercises:
            if st.button(ex["name"], key=f"btn_{ex['id']}"):
                st.session_state.selected_exercise = ex['id']
                st.success(f"{ex['name']} 선택됨")
        
        st.divider()
        
        # 운동 정보 표시
        if st.session_state.selected_exercise:
            selected_ex = next((ex for ex in exercises if ex['id'] == st.session_state.selected_exercise), None)
            if selected_ex:
                st.write(f"**현재 운동**: {selected_ex['name']}")
                st.write(f"**설명**: {selected_ex['description']}")
        
        # 운동 기록 표시
        if st.session_state.accuracy_history:
            st.subheader("📊 운동 통계")
            avg_accuracy = np.mean(st.session_state.accuracy_history)
            st.metric("평균 정확도", f"{avg_accuracy:.1f}%")
    
    # 카메라 시뮬레이션 루프
    if camera_on:
        # 더미 이미지 표시
        frame = generate_dummy_image()
        cam_placeholder.image(frame, channels="RGB", use_container_width=True)
        
        # 더미 이미지 전송 (실제 자세 데이터 수신을 위해)
        if st.session_state.selected_exercise:
            result = send_image(st.session_state.selected_exercise)
        
        # 수신된 자세 데이터 표시
        if st.session_state.ws_data['latest_pose']:
            # 자세 정확도 표시
            accuracy = st.session_state.ws_data['accuracy']
            if accuracy > 0:
                accuracy_gauge.metric("자세 정확도", f"{accuracy:.1f}%")
        
        # 정확도 차트 표시
        if st.session_state.accuracy_history:
            chart_area.line_chart(st.session_state.accuracy_history)
        
        # 시뮬레이션을 위한 딜레이
        time.sleep(0.5)

if __name__ == "__main__":
    main() 