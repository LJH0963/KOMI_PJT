import cv2
import tkinter as tk
from tkinter import filedialog
import os
import time
from ultralytics import YOLO
import json
from PIL import Image
import numpy as np

# TKinter GUI 숨기기
root = tk.Tk()
root.withdraw()

# 결과 저장 폴더 선택
output_folder = filedialog.askdirectory(title='결과 저장 폴더를 선택하세요')
if not output_folder:
    print("결과를 저장할 폴더를 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# 저장용 폴더 생성
video_path = os.path.join(output_folder, 'recorded_video.avi')
image_output_folder = os.path.join(output_folder, 'image')
json_output_folder = os.path.join(output_folder, 'json')
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(json_output_folder, exist_ok=True)

# 기준 pose json 로딩 함수
def load_reference_pose(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints = []
    for kp in data['keypoints']:
        if kp["x"] is not None and kp["y"] is not None:
            keypoints.append([kp["x"], kp["y"]])
        else:
            keypoints.append([None, None])
    return np.array(keypoints, dtype=np.float32)

# 거리 기반 유사성 판단 함수
def is_pose_similar(current_pose, reference_pose, threshold=50):
    if current_pose is None or reference_pose is None:
        return False
    valid_indices = [
        i for i in range(len(reference_pose))
        if reference_pose[i][0] is not None and current_pose[i][0] is not None
    ]
    if not valid_indices:
        return False
    diffs = [
        np.linalg.norm(current_pose[i] - reference_pose[i])
        for i in valid_indices
    ]
    mean_dist = np.mean(diffs)
    return mean_dist < threshold

# YOLO 모델 로딩
yolo_model = YOLO("yolo11n-pose.pt")

# 기준 마스크 이미지 및 keypoint 로딩
mask_image_path = 'C:/Users/user/Desktop/data/frame100_mask_rgba.png'
mask = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
if mask is None:
    print("마스크 이미지를 찾을 수 없습니다.")
    exit()
reference_pose = load_reference_pose("C:/Users/user/Desktop/img_output/squat/front_json/json/frame100.json")

# 마스크 오버레이 함수
def overlay_mask(frame, mask):
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    alpha_mask = mask_resized[:, :, 3] / 255.0
    for c in range(3):
        frame[:, :, c] = frame[:, :, c] * (1 - alpha_mask) + mask_resized[:, :, c] * alpha_mask
    return frame

# 마스크 기반 정렬 판단 함수
def is_pose_aligned(keypoints, mask, threshold_ratio=0.3):
    mask_area = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask_area, 10, 255, cv2.THRESH_BINARY)
    h, w = binary_mask.shape
    inside_count = 0
    total_count = 0
    for x, y in keypoints:
        if x is not None and y is not None:
            px, py = int(x), int(y)
            if 0 <= px < w and 0 <= py < h:
                if binary_mask[py, px] > 0:
                    inside_count += 1
                total_count += 1
    if total_count == 0:
        return False
    return (inside_count / total_count) >= threshold_ratio

# 웹캠 설정 및 비디오 객체 생성
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))
print("요청한 FPS: 30, 실제 적용된 FPS:", cap.get(cv2.CAP_PROP_FPS))

# 정렬 대기 루프
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
    vis_frame = frame.copy()
    vis_frame = overlay_mask(vis_frame, mask)
    if keypoints is not None:
        keypoints_array = np.array(keypoints, dtype=np.float32)
        if is_pose_aligned(keypoints_array, mask) or is_pose_similar(keypoints_array, reference_pose):
            cv2.putText(vis_frame, "Pose Aligned! Starting soon...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            aligned = True
            time.sleep(1)
        else:
            cv2.putText(vis_frame, "Align your pose with the mask", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(vis_frame, "Detecting pose...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.imshow("Pose Alignment", vis_frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

# 카운트다운
print("정렬 완료! 5초 후 녹화가 시작됩니다.")
countdown_start = time.time()
while time.time() - countdown_start < 5:
    remaining = 5 - int(time.time() - countdown_start)
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    cv2.putText(frame, str(remaining), (600, 380), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
    cv2.imshow('Recording Video', frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

# 2.9초간 녹화
record_start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    if time.time() - record_start_time <= 2.9:
        out.write(frame)
    else:
        print("2.9초 녹화 완료. 영상 저장 종료.")
        break
    cv2.imshow('Recording Video', frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"녹화 완료: {video_path}")

# 87프레임 추출
cap = cv2.VideoCapture(video_path)
saved_idx = 0
frame_count = 0
while cap.isOpened() and frame_count < 87:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    image_name = f"frame_{saved_idx:04d}.jpg"
    output_image_path = os.path.join(image_output_folder, image_name)
    cv2.imwrite(output_image_path, frame)
    print(f"프레임 저장: {image_name}")
    saved_idx += 1
    frame_count += 1
cap.release()
cv2.destroyAllWindows()
print("모든 프레임 추출 및 저장 완료.")

# YOLO-Pose 적용 및 JSON 저장
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
image_files = sorted([f for f in os.listdir(image_output_folder) if f.endswith('.jpg')])
for image_file in image_files:
    image_path = os.path.join(image_output_folder, image_file)
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
    json_output_path = os.path.join(json_output_folder, image_file.replace('.jpg', '.json'))
    with open(json_output_path, 'w', encoding='utf-8') as jf:
        json.dump(json_data, jf, indent=4)
    print(f"YOLO-Pose 완료 및 JSON 저장: {json_output_path}")
