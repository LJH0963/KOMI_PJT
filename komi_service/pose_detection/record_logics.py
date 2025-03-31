import cv2
import tkinter as tk
from tkinter import filedialog
import os
import time
from ultralytics import YOLO
import json
import numpy as np
import subprocess
import shutil

def select_output_folder():
    """결과 저장 폴더를 선택하는 함수"""
    root = tk.Tk()
    root.withdraw()
    output_folder = filedialog.askdirectory(title='결과 저장 폴더를 선택하세요')
    if not output_folder:
        return None
    return output_folder

def create_folder_structure(output_folder):
    """결과 저장을 위한 폴더 구조를 생성하는 함수"""
    # 직접 H.264 코덱을 사용한 MP4 파일 경로
    video_path = os.path.join(output_folder, 'recorded_video.mp4')
    image_output_folder = os.path.join(output_folder, 'image')
    json_output_folder = os.path.join(output_folder, 'json')
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(json_output_folder, exist_ok=True)
    return video_path, image_output_folder, json_output_folder

def load_reference_pose(json_path):
    """기준 포즈 JSON을 로드하는 함수"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints = []
    for kp in data['keypoints']:
        if kp["x"] is not None and kp["y"] is not None:
            keypoints.append([kp["x"], kp["y"]])
        else:
            keypoints.append([None, None])
    return np.array(keypoints, dtype=np.float32)

def is_pose_similar_by_accuracy(current_pose, reference_pose, threshold_px=20, ratio=0.7):
    """정확도 기반 포즈 유사성 판단 함수"""
    if current_pose is None or reference_pose is None:
        return False
    match_count = 0
    total_count = 0
    for i in range(len(reference_pose)):
        ref = reference_pose[i]
        cur = current_pose[i]
        if ref[0] is not None and cur[0] is not None:
            dist = np.linalg.norm(np.array(ref) - np.array(cur))
            total_count += 1
            if dist <= threshold_px:
                match_count += 1
    if total_count == 0:
        return False
    return (match_count / total_count) >= ratio

def load_yolo_model(model_path="yolo11x-pose.pt"):
    """YOLO 모델을 로드하는 함수"""
    return YOLO(model_path)

def overlay_mask(frame, mask, alpha_value=100):
    """마스크 오버레이 함수 (반투명 적용)"""
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask_rgb = mask_resized[:, :, :3].astype(np.uint8)
    mask_alpha = mask_resized[:, :, 3].astype(np.uint8)
    object_mask = (mask_alpha > 0).astype(np.uint8)
    custom_alpha = np.full_like(mask_alpha, alpha_value, dtype=np.uint8)
    custom_alpha[object_mask == 0] = 0
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    for c in range(3):
        frame_rgba[:, :, c] = (
            frame_rgba[:, :, c] * (1 - custom_alpha / 255.0) +
            mask_rgb[:, :, c] * (custom_alpha / 255.0)
        ).astype(np.uint8)
    frame_rgba[:, :, 3] = np.maximum(frame_rgba[:, :, 3], custom_alpha)
    return cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

def setup_camera(width=1280, height=720, fps=30):
    """웹캠 설정 함수"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

def setup_video_writer(video_path, width=1280, height=720, fps=30):
    """비디오 녹화 설정 함수 - H.264 코덱으로 직접 저장"""
    # H.264 코덱으로 직접 저장 (mp4 확장자)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 코덱
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # VideoWriter 객체가 제대로 생성되었는지 확인
    if not out.isOpened():
        print("H.264 코덱으로 비디오 녹화 설정 실패. 대안 시도 중...")
        # 일부 시스템에서는 다른 코덱 문자열이 필요할 수 있음
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("H.264 코덱 사용 불가. XVID로 대체합니다.")
            # H.264를 사용할 수 없는 경우 XVID로 대체
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # 확장자를 avi로 변경
            video_path = video_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    return out, video_path

def convert_video_to_h264(input_path, output_path):
    """녹화된 비디오를 H.264 코덱으로 변환하는 함수"""
    try:
        # ffmpeg를 사용하여 H.264 코덱으로 변환
        cmd = [
            'ffmpeg', '-y', '-i', input_path, 
            '-c:v', 'libx264', '-preset', 'medium', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"비디오가 H.264 코덱으로 성공적으로 변환되었습니다: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"비디오 변환 중 오류 발생: {e}")
        return False
    except FileNotFoundError:
        print("FFmpeg가 설치되어 있지 않습니다. FFmpeg를 설치하거나 경로를 확인하세요.")
        return False

def wait_for_pose_alignment(cap, yolo_model, reference_pose, mask_image):
    """포즈 정렬 대기 함수"""
    print("초기 정렬 대기 중...")
    aligned = False
    while not aligned:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        results = yolo_model.predict(source=frame, stream=False, verbose=False)
        keypoints = None
        for result in results:
            if result.keypoints is not None:
                keypoints_np = result.keypoints.xy.cpu().numpy()
                keypoints = keypoints_np[0]
                break
        # vis_frame = frame.copy()
        # vis_frame = overlay_mask(vis_frame, mask_image)
        if keypoints is not None:
            keypoints_array = np.array(keypoints, dtype=np.float32)
            if is_pose_similar_by_accuracy(keypoints_array, reference_pose):
                # cv2.putText(vis_frame, "Pose Aligned! Starting soon...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                aligned = True
                time.sleep(1)
            # else:
                # cv2.putText(vis_frame, "Align your pose with the reference", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # else:
            # cv2.putText(vis_frame, "Detecting pose...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        # cv2.imshow("Pose Alignment", vis_frame)
        # if cv2.waitKey(1) == 27:
        #     cv2.destroyAllWindows()
        #     return False
    return True

def countdown_timer(cap, seconds=5):
    """카운트다운 타이머 함수"""
    print(f"{seconds}초 후 녹화가 시작됩니다.")
    countdown_start = time.time()
    while time.time() - countdown_start < seconds:
        remaining = seconds - int(time.time() - countdown_start)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        cv2.putText(frame, str(remaining), (600, 380), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
        cv2.imshow('Recording Video', frame)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return False
    return True

def record_video(cap, out, duration=2.9):
    """비디오 녹화 함수"""
    print(f"{duration}초 동안 녹화합니다.")
    record_start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        if time.time() - record_start_time <= duration:
            out.write(frame)
        else:
            print(f"{duration}초 녹화 완료. 영상 저장 종료.")
            break
        cv2.imshow('Recording Video', frame)
        if cv2.waitKey(1) == 27:
            break
    return True

def extract_frames(video_path, output_folder, num_frames=87):
    """녹화된 비디오에서 프레임 추출 함수"""
    cap = cv2.VideoCapture(video_path)
    saved_idx = 0
    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image_name = f"frame_{saved_idx:03d}.jpg"
        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, frame)
        saved_idx += 1
        frame_count += 1
    cap.release()
    print(f"{saved_idx}개 프레임 추출 및 저장 완료.")
    return saved_idx

def detect_and_save_pose_keypoints(yolo_model, image_folder, json_folder):
    """포즈 감지 및 JSON 저장 함수"""
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        results = yolo_model(image)
        json_data = {'image_name': image_file, 'bboxes': [], 'keypoints': []}
        keypoints_dict = {part: {"x": None, "y": None, "confidence": 0.0} for part in COCO_KEYPOINTS}
        
        for result in results:
            if result.boxes is not None:
                bboxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for bbox, conf, cls in zip(bboxes, confs, classes):
                    x1, y1, x2, y2 = map(int, bbox)
                    json_data['bboxes'].append({
                        'class': int(cls),
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf)
                    })
            
            keypoints = result.keypoints.xy.cpu().numpy()
            scores = result.keypoints.conf.cpu().numpy()
            for idx, (kp, score) in enumerate(zip(keypoints[0], scores[0])):
                x, y = int(kp[0]), int(kp[1])
                conf = float(score)
                keypoints_dict[COCO_KEYPOINTS[idx]] = {
                    'x': x if conf > 0.1 else None,
                    'y': y if conf > 0.1 else None,
                    "confidence": conf
                }
        
        json_data["keypoints"] = [
            {
                "part": part,
                "x": keypoints_dict[part]["x"],
                "y": keypoints_dict[part]["y"],
                "confidence": keypoints_dict[part]["confidence"]
            } for part in COCO_KEYPOINTS
        ]
        
        json_output_path = os.path.join(json_folder, image_file.replace('.jpg', '.json'))
        with open(json_output_path, 'w', encoding='utf-8') as jf:
            json.dump(json_data, jf, indent=4)
    
    print(f"모든 프레임 포즈 감지 및 JSON 저장 완료.")
    return len(image_files)

def load_mask_image(mask_image_path):
    """마스크 이미지 로드 함수"""
    mask = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("마스크 이미지를 찾을 수 없습니다.")
        return None
    return mask

def main(reference_json_path, mask_image_path):
    """메인 처리 함수"""
    # 결과 저장 폴더 선택
    output_folder = "C:/wanted/KOMI_PJT/data/test"
    
    # 폴더 구조 생성 (H.264 비디오 파일 경로)
    video_path, image_folder, json_folder = create_folder_structure(output_folder)
    
    # YOLO 모델 로드
    yolo_model = load_yolo_model()
    
    # 마스크 이미지 로드
    mask_image = load_mask_image(mask_image_path)
    if mask_image is None:
        return False
    
    # 기준 포즈 로드
    reference_pose = load_reference_pose(reference_json_path)
    
    # 카메라 설정
    cap = setup_camera()
    
    # 비디오 녹화 설정 (H.264 직접 사용)
    out, video_path = setup_video_writer(video_path)
    
    # 포즈 정렬 대기
    if not wait_for_pose_alignment(cap, yolo_model, reference_pose, mask_image):
        cap.release()
        out.release()
        return False
    
    # 카운트다운
    if not countdown_timer(cap):
        cap.release()
        out.release()
        return False
    
    # 비디오 녹화 (H.264 직접 사용)
    record_video(cap, out)
    
    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # H.264 코덱으로 저장된 비디오에서 프레임 추출
    extract_frames(video_path, image_folder)
    
    # 포즈 감지 및 JSON 저장
    detect_and_save_pose_keypoints(yolo_model, image_folder, json_folder)
    
    print("모든 처리가 완료되었습니다.")
    return True
