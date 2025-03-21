import cv2
import tkinter as tk
from tkinter import filedialog
import os
import time
from ultralytics import YOLO
import json

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

# 1단계: 웹캠 영상 저장 (1280x720, 실시간 녹화)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# FPS 설정 확인
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"요청한 FPS: 30, 실제 적용된 FPS: {actual_fps}")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

print("웹캠 실행 완료. 5초 후부터 2.9초간 영상만 저장합니다.")

# 카운트다운 표시
countdown_start = time.time()
while time.time() - countdown_start < 5:
    remaining = 5 - int(time.time() - countdown_start)
    ret, frame = cap.read()
    if not ret:
        break
    countdown_text = str(remaining)
    cv2.putText(frame, countdown_text, (600, 380), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
    cv2.imshow('Recording Video', frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

# 본격적인 녹화 시작 (2.9초간)
save_started = True
record_start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
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

# 2단계: 저장된 영상에서 정확히 87프레임 추출 (30fps * 2.9초)
cap = cv2.VideoCapture(video_path)
saved_idx = 0
frame_count = 0

while cap.isOpened() and frame_count < 87:
    ret, frame = cap.read()
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

# 3단계: 저장된 프레임에 대해 YOLO-Pose 적용 및 JSON 저장
yolo_model = YOLO("yolo11x-pose.pt")
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
