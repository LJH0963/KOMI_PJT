import streamlit as st
import time
import queue

from frontend import utils
from frontend import config
from frontend import camera
from frontend import websocket
from frontend.session import app_state, go_to_exercise_view, go_to_exercise_select

# 영상 표시 함수 - 별도 함수로 분리
def display_exercise_videos(exercise):
    """운동 가이드 영상을 표시하는 함수"""
    try:
        st.header(f"{exercise['name']} 가이드 영상")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("정면 뷰")
            front_video_url = f"{config.API_URL}/video{exercise['guide_videos']['front']}"
            # 자동 재생 및 반복 재생을 위한 HTML 코드 사용
            front_video_html = f"""
            <video autoplay loop muted width="100%" controls>
                <source src="{front_video_url}" type="video/mp4">
                브라우저가 비디오 태그를 지원하지 않습니다.
            </video>
            """
            st.components.v1.html(front_video_html, height=300)
        
        with col2:
            st.subheader("측면 뷰")
            side_video_url = f"{config.API_URL}/video{exercise['guide_videos']['side']}"
            # 자동 재생 및 반복 재생을 위한 HTML 코드 사용
            side_video_html = f"""
            <video autoplay loop muted width="100%" controls>
                <source src="{side_video_url}" type="video/mp4">
                브라우저가 비디오 태그를 지원하지 않습니다.
            </video>
            """
            st.components.v1.html(side_video_html, height=300)
    except Exception as e:
        st.error(f"영상을 표시하는 중 오류가 발생했습니다: {str(e)}")

# 운동 선택 페이지
def exercise_select_page():
    st.title("KOMI 운동 가이드")
    
    # 사이드바에 운동 가이드 버튼 (현재 페이지이므로 비활성화)
    with st.sidebar:
        st.header("메뉴")
        st.button("운동 가이드", disabled=True, use_container_width=True)
        
        # 영상 표시 상태일 때만 운동하기 버튼 표시
        if st.session_state.show_videos:
            st.button("운동하기", on_click=go_to_exercise_view, type="primary", use_container_width=True)
                
    # 영상 보기 상태이면 영상 표시
    if st.session_state.show_videos and st.session_state.selected_exercise:
        display_exercise_videos(st.session_state.selected_exercise)
            
    # 운동 목록 가져오기
    exercises_data = utils.run_async(camera.get_exercises())
    if not exercises_data or "exercises" not in exercises_data:
        st.error("운동 목록을 가져올 수 없습니다. 서버 연결을 확인해주세요.")
        return
    
    exercises = exercises_data["exercises"]
    
    # 운동 선택 UI
    st.header("운동 선택")
    selected_exercise = st.selectbox(
        "운동 선택",
        options=exercises,
        format_func=lambda x: f"{x['name']} ({x['difficulty']})",
        key="exercise_select"
    )
    
    if selected_exercise:
        st.session_state.selected_exercise = selected_exercise
        st.write(f"**설명:** {selected_exercise['description']}")
        st.write(f"**난이도:** {selected_exercise['difficulty']}")
        
        # 영상 보기/숨기기 버튼
        button_text = "영상 숨기기" if st.session_state.show_videos else "영상 보기"
        button_type = "secondary" if st.session_state.show_videos else "primary"
        
        if st.button(button_text, key="toggle_video_btn", type=button_type):
            st.session_state.show_videos = not st.session_state.show_videos
            st.rerun()

# 운동하기 페이지
def exercise_view_page():
    if not st.session_state.selected_exercise:
        st.warning("선택된 운동이 없습니다. 운동을 선택해주세요.")
        st.button("운동 선택으로 돌아가기", on_click=go_to_exercise_select)
        return
    
    exercise = st.session_state.selected_exercise
    
    # 사이드바에 운동 가이드 버튼
    with st.sidebar:
        st.header("메뉴")
        st.button("운동 가이드", on_click=go_to_exercise_select, use_container_width=True)
    
    st.title(f"{exercise['name']} 운동")
    st.caption(f"난이도: {exercise['difficulty']}")
    
    # 서버 상태 확인
    if st.session_state.server_status is None:
        with st.spinner("서버 연결 중..."):
            cameras, status = camera.get_cameras()
            st.session_state.cameras = cameras
            st.session_state.server_status = status
    
    # 서버 상태에 따른 처리
    if st.session_state.server_status == "연결 실패":
        st.error("서버에 연결할 수 없습니다")
        if st.button("재연결"):
            cameras, status = camera.get_cameras()
            st.session_state.cameras = cameras
            st.session_state.server_status = status
            st.rerun()
        return
    
    # 카메라 목록이 없으면 표시
    if not st.session_state.cameras:
        st.info("연결된 카메라가 없습니다")
        if st.button("새로고침"):
            cameras, status = camera.get_cameras()
            st.session_state.cameras = cameras
            st.session_state.server_status = status
            st.rerun()
        return
        
    # 카메라 선택 UI
    st.header("카메라 선택")
    col1, col2 = st.columns(2)
    
    with col1:
        front_camera = st.selectbox(
            "프론트 카메라 선택",
            ["선택 안함"] + st.session_state.cameras,
            key="front_camera_select",
            index=0 if st.session_state.front_camera == "선택 안함" else 
                   ["선택 안함"] + st.session_state.cameras.index(st.session_state.front_camera) + 1
        )
        st.session_state.front_camera = front_camera
    
    with col2:
        # 프론트 카메라를 선택한 경우, 해당 카메라를 사이드 카메라 목록에서 제외
        available_side_cameras = ["선택 안함"] + [
            cam for cam in st.session_state.cameras 
            if front_camera == "선택 안함" or cam != front_camera
        ]
        side_camera = st.selectbox(
            "사이드 카메라 선택",
            available_side_cameras,
            key="side_camera_select",
            index=0 if st.session_state.side_camera == "선택 안함" or st.session_state.side_camera not in available_side_cameras else 
                   available_side_cameras.index(st.session_state.side_camera)
        )
        st.session_state.side_camera = side_camera
    
    # 선택된 카메라 업데이트 - 선택된 카메라만 포함
    selected_cameras_list = []
    if st.session_state.front_camera != "선택 안함":
        selected_cameras_list.append(st.session_state.front_camera)
    if st.session_state.side_camera != "선택 안함":
        selected_cameras_list.append(st.session_state.side_camera)
    
    if selected_cameras_list:
        st.session_state.selected_cameras = selected_cameras_list
        app_state.selected_cameras = selected_cameras_list
        # 카메라 선택 변경 시 동기화 버퍼 초기화
        camera.init_sync_buffer(selected_cameras_list)
    
    # 포즈 표시 체크박스 제거 (포즈 표시는 항상 활성화)
    st.session_state.show_pose_overlay = True
    
    if len(selected_cameras_list) == 0:
        st.warning("선택된 카메라가 없습니다. 카메라를 선택해주세요.")
        return
    
    # 웹캠 화면 표시
    cols = st.columns(2)
    image_slots = {}
    status_slots = {}
    connection_indicators = {}
    pose_status_slots = {}
    
    # 프론트 카메라
    with cols[0]:
        st.subheader("프론트 뷰")
        if st.session_state.front_camera != "선택 안함":
            connection_indicators[st.session_state.front_camera] = st.empty()
            image_slots[st.session_state.front_camera] = st.empty()
            status_slots[st.session_state.front_camera] = st.empty()
            pose_status_slots[st.session_state.front_camera] = st.empty()
            status_slots[st.session_state.front_camera].text("실시간 스트리밍 준비 중...")
        else:
            st.info("프론트 카메라를 선택해주세요.")
    
    # 사이드 카메라
    with cols[1]:
        st.subheader("사이드 뷰")
        if st.session_state.side_camera != "선택 안함":
            connection_indicators[st.session_state.side_camera] = st.empty()
            image_slots[st.session_state.side_camera] = st.empty()
            status_slots[st.session_state.side_camera] = st.empty()
            pose_status_slots[st.session_state.side_camera] = st.empty()
            status_slots[st.session_state.side_camera].text("실시간 스트리밍 준비 중...")
        else:
            st.info("사이드 카메라를 선택해주세요.")
    
    # 별도 스레드 시작 (단 한번만)
    if not st.session_state.thread_started:
        websocket.start_websocket_thread()
        st.session_state.thread_started = True
    
    # 동기화 상태 표시
    st.text(st.session_state.sync_status)
    
    # 메인 UI 업데이트 루프
    update_interval = 0
    placeholder = st.empty()
    
    with placeholder.container():
        while True:
            update_interval += 1
            update_ui = False
            
            if len(st.session_state.selected_cameras) > 1:
                # 동기화된 프레임 찾기
                sync_frames, sync_status = camera.find_synchronized_frames(st.session_state.selected_cameras)
                if sync_frames:
                    # 동기화된 프레임이 있으면 UI 업데이트
                    for camera_id, frame_data in sync_frames.items():
                        st.session_state.camera_images[camera_id] = frame_data["image"]
                        st.session_state.image_update_time[camera_id] = frame_data["time"]
                    
                    # 동기화 상태 업데이트
                    if sync_status:
                        st.session_state.sync_status = sync_status
                        
                    update_ui = True
            else:
                # 동기화 없이 각 카메라의 최신 프레임 사용
                for camera_id in st.session_state.selected_cameras:
                    if camera_id in app_state.image_queues and not app_state.image_queues[camera_id].empty():
                        try:
                            img_data = app_state.image_queues[camera_id].get(block=False)
                            st.session_state.camera_images[camera_id] = img_data.get("image")
                            st.session_state.image_update_time[camera_id] = img_data.get("time")
                            update_ui = True
                        except queue.Empty:
                            pass
            
            # 이미지 업데이트
            if update_ui:
                for camera_id in st.session_state.selected_cameras:
                    if camera_id in st.session_state.camera_images and camera_id in image_slots:
                        img = st.session_state.camera_images[camera_id]
                        if img is not None:
                            # 포즈 오버레이가 활성화되고, 해당 카메라의 포즈 데이터가 있는 경우 오버레이
                            if st.session_state.show_pose_overlay and camera_id in app_state.pose_data_store:
                                # 포즈 데이터로 이미지 그리기
                                img = utils.draw_pose_on_image(img, app_state.pose_data_store[camera_id])
                                
                                # 포즈 상태 업데이트
                                if camera_id in app_state.pose_update_times:
                                    pose_time = app_state.pose_update_times[camera_id].strftime('%H:%M:%S.%f')[:-3]
                                    pose_status_slots[camera_id].text(f"포즈 업데이트: {pose_time}")
                            else:
                                pose_status_slots[camera_id].text("")
                            
                            # 이미지 표시
                            image_slots[camera_id].image(img, use_container_width=True)
                            status_time = st.session_state.image_update_time[camera_id].strftime('%H:%M:%S.%f')[:-3]
                            status_slots[camera_id].text(f"업데이트: {status_time}")
            
            # UI 업데이트 간격 (더 빠른 응답성)
            time.sleep(0.05)
            
            # 10초마다 세션 상태 체크 - 페이지 전환됐을 경우 루프 중지
            if update_interval % 200 == 0:  # 0.05 * 200 = 10초
                if st.session_state.page != 'exercise_view':
                    break 