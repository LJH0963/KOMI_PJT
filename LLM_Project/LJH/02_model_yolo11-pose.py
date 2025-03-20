import cv2
import tkinter as tk
from tkinter import filedialog
import os
from ultralytics import YOLO
import json

# TKinter의 GUI 숨기기
root = tk.Tk()
root.withdraw()

# 이미지 폴더 먼저 선택
image_folder = filedialog.askdirectory(title="이미지 폴더를 선택하세요")

# 이미지가 없을 경우에 대한 설정
if not image_folder:
    print("폴더 내 이미지 파일이 없습니다. 프로그램을 종료합니다.")
    exit()

# 결과 저장 폴더 선택
output_folder = filedialog.askdirectory(title='저장될 폴더를 선택하세요')

# 출력 폴더를 설정하지 않을 경우?
if not output_folder:
    print("결과를 저장할 폴더를 선택하지 않았습니다. 프로그램을 종료할겁니다.")
    exit()

# YOLO-Pose 모델 로드
yolo_model = YOLO("yolo11x-pose.pt")

# 이미지 확장자 지정
## trouble : list로 넣으니 문제가 생김 -> tuple로 변경
image_extensions = (".jpg", ".jpeg", ".png")

# 폴더 내의 모든 이미지 파일 찾기
image_paths = []
for root_dir, _, files in os.walk(image_folder):       # 현재 경로 / 하위 폴더 리스트(사용 X)/ 폴더 내 모든 파일
    for file in files:
        if file.lower().endswith(image_extensions):
            image_paths.append(os.path.join(root_dir, file))

# 처리할 이미지가 없는 경우 = 경로 내에서 수행이 안될 경우
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
    # 이미지 파일명 추출
    image_name = os.path.basename(image_path)
    image_name_no_ext = os.path.splitext(image_name)[0]
    
    # 이미지 로드
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"{image_name}을 불러올 수 없습니다.")
        continue

    # YOLO-Pose 모델을 사용하여 포즈 감지
    results = yolo_model(image)

    # Json 파일 저장을 위한 빈 리스트 생성
    json_data = {'image_name':image_name, 'keypoints':[]}

    # 관절(Keypoints) 좌표 추출 및 시각화
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()  # 좌표 변환
        scores = result.keypoints.conf.cpu().numpy()  # 신뢰도 변환

        for kp, score in zip(keypoints[0], scores[0]):
                x, y = int(kp[0]), int(kp[1])
                conf = float(score)

                json_data['keypoints'].append({"x": x, "y": y, "confidence": conf})

                if conf > 0.5:
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 빨간색 점으로 표시

    # 결과 이미지 저장
    output_image_path = os.path.join(image_output_folder, f"result_{image_name}")
    cv2.imwrite(output_image_path, image)
    print(f"결과 이미지 저장 완료: {output_image_path}")

    # Json 저장
    json_file_path = os.path.join(json_output_folder, f"{image_name_no_ext}.json")
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump({"keypoints": json_data}, json_file, indent=4)
    print(f"Json 저장완료 : {json_file_path}")