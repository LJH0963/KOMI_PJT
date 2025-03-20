import cv2
import tkinter as tk
from tkinter import filedialog
import os
import torch
import json
from ultralytics import YOLO

# TKinter의 GUI 숨기기
root = tk.Tk()
root.withdraw()

# 이미지 폴더 선택
image_folder = filedialog.askdirectory(title="이미지 폴더를 선택하세요")

if not image_folder:
    print("폴더 내 이미지 파일이 없습니다. 프로그램을 종료합니다.")
    exit()

# 결과 저장 폴더 선택
output_folder = filedialog.askdirectory(title='저장될 폴더를 선택하세요')

if not output_folder:
    print("결과를 저장할 폴더를 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📌 Using device: {device}")

# YOLO-Pose 모델 로드 (GPU 적용)
yolo_model = YOLO("yolo11x-pose.pt").to(device)

# COCO Keypoint 이름 리스트
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO 데이터셋 기준의 관절 연결 정보
skeleton = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # 팔 (오른쪽, 왼쪽)
    (11, 13), (13, 15), (12, 14), (14, 16),  # 다리 (오른쪽, 왼쪽)
    (5, 6), (11, 12), (5, 11), (6, 12)  # 몸통 연결
]

# 이미지 확장자 지정
image_extensions = (".jpg", ".jpeg", ".png")

# 폴더 내의 모든 이미지 파일 찾기
image_paths = []
for root_dir, _, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_paths.append(os.path.join(root_dir, file))

if not image_paths:
    print("선택한 폴더 내에 처리할 이미지가 없습니다.")
    exit()

# 결과 이미지 저장 폴더 생성
image_output_folder = os.path.join(output_folder, 'image')
json_output_folder = os.path.join(output_folder, 'json')
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(json_output_folder, exist_ok=True)

# 이미지 처리
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    image_name_no_ext = os.path.splitext(image_name)[0]
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"{image_name}을 불러올 수 없습니다.")
        continue

    # YOLO-Pose 모델을 사용하여 포즈 감지 (GPU 적용)
    results = yolo_model(image)

    # JSON 데이터 저장을 위한 초기화
    json_data = {
        'image_name': image_name,
        'bboxes': [],   # Bounding Box 정보 저장
        'keypoints': [] # Keypoints 정보 저장
    }

    # 17개 부위 전부 초기화
    keypoints_dict = {part: {"x": None, "y": None, "confidence": 0.0} for part in COCO_KEYPOINTS}

    for result in results:
        # Bounding Box 정보 가져오기
        if result.boxes is not None:
            bboxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for bbox, conf, cls in zip(bboxes, confs, classes):
                x1, y1, x2, y2 = map(int, bbox)
                json_data["bboxes"].append({
                    "class": int(cls),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf)
                })
                # Bounding Box 시각화
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색 박스
                cv2.putText(image, f"Conf: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Keypoint 정보 가져오기
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
        scores = result.keypoints.conf.cpu().numpy() if result.keypoints is not None else []

        if len(keypoints) > 0:
            for idx, (kp, score) in enumerate(zip(keypoints[0], scores[0])):
                x, y = int(kp[0]), int(kp[1])
                conf = float(score)

                keypoints_dict[COCO_KEYPOINTS[idx]] = {
                    'x': x if conf > 0.1 else None,
                    'y': y if conf > 0.1 else None,
                    "confidence": conf
                }

                if conf > 0.5:
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 빨간색 점으로 표시

            # 관절 연결선 그리기
            for joint1, joint2 in skeleton:
                part1 = COCO_KEYPOINTS[joint1]
                part2 = COCO_KEYPOINTS[joint2]
                kp1 = keypoints_dict[part1]
                kp2 = keypoints_dict[part2]

                if kp1['confidence'] > 0.5 and kp2['confidence'] > 0.5:
                    cv2.line(image, (kp1['x'], kp1['y']), (kp2['x'], kp2['y']), (0, 255, 0), 2)

    # JSON에 Keypoints 추가
    json_data["keypoints"] = [
        {
            "part": part,
            "x": keypoints_dict[part]["x"],
            "y": keypoints_dict[part]["y"],
            "confidence": keypoints_dict[part]["confidence"]
        } for part in COCO_KEYPOINTS
    ]

    # 결과 이미지 저장
    output_image_path = os.path.join(image_output_folder, f"result_{image_name}")
    cv2.imwrite(output_image_path, image)
    print(f"✅ 결과 이미지 저장 완료: {output_image_path}")

    # JSON 저장
    json_file_path = os.path.join(json_output_folder, f"{image_name_no_ext}.json")
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"✅ Json 저장 완료: {json_file_path}")
