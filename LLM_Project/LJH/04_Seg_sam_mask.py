import cv2
import numpy as np
import torch
import os
import glob
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from tkinter import filedialog

# 입력 및 출력 디렉토리 설정
input_dir = "C:/Users/user/Desktop/img_output/squat/front"  # 입력 이미지 폴더
output_dir = "C:/Users/user/Desktop/img_output/squat/mask"  # 세그멘테이션 결과 저장 폴더
os.makedirs(output_dir, exist_ok=True)  # 출력 폴더 생성

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLO11 모델 로드
model = YOLO('yolo11x.pt')

# SAM 모델 로드
sam_checkpoint = "C:/WANTED/LLM/KOMI_PJT/tests/LJH/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 입력 폴더 내 모든 jpg 파일 찾기
image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

# 파일이 없을 경우 경고
if not image_paths:
    print("경고: 해당 디렉토리에 JPG 파일이 없습니다!")

# 모든 이미지에 대해 반복 처리
for image_path in image_paths:
    # 원본 이미지 파일 이름 가져오기
    image_name = os.path.basename(image_path).split(".")[0]  # 확장자 제거
    output_path = os.path.join(output_dir, f"{image_name}_mask.png")  # 저장 경로 설정

    print(f"처리 중: {image_path} -> {output_path}")

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"오류: {image_path} 를 불러올 수 없습니다.")
        continue

    # 객체 검출 수행
    results = model.predict(source=image, conf=0.6)

    # 검출된 경계 상자 추출
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    # 이미지 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # SAM에 이미지 설정
    predictor.set_image(image_rgb)

    # 경계 상자를 사용하여 세그멘테이션 마스크 생성
    transformed_boxes = predictor.transform.apply_boxes_torch(torch.tensor(bboxes, dtype=torch.float32), image_rgb.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False
    )

    # # 결과 저장을 위한 빈 마스크 생성 (흰색 배경)
    # segmentation_result = np.ones_like(image_rgb[:, :, 0]) * 255  # 흰색 (255)
    ## Trouble shooting : Web_yolo 사용 시 png 포맷이 필요하나, 기본적으로 jpg 저장되는 모델이므로 해당 내용 변경
    ### 위쪽에 mask.png로 저장되도록 하고, RGBA 마스크로 초기화한다.
    
    # RGBA 마스크 초기화(흰색 배경 + 알파 채널 0)
    rgba_mask = np.ones((image.shape[0], image.shape[1], 4), dtype=np.uint8) * 255
    rgba_mask[:, :, 3] = 0  # 기본 알파는 투명으로 설정함.

    # 객체 영역, 즉 마스크가 될 부분은 255로 검은색으로 변경
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for mask in masks:
        mask = mask.cpu().numpy().astype(np.uint8).squeeze()  # 차원 축소 후 적용
        combined_mask[mask > 0] = 1  # 객체 부분을 검은색(0)으로 설정

    # 객체 영역만 RGB를 검정으로, 알파를 255로 설정
    rgba_mask[combined_mask == 1, 0] = 0  # R
    rgba_mask[combined_mask == 1, 1] = 0  # G
    rgba_mask[combined_mask == 1, 2] = 0  # B
    rgba_mask[combined_mask == 1, 3] = 255  # A

    # 결과 저장
    cv2.imwrite(output_path, rgba_mask)
    print(f"✅ 저장 완료: {output_path}")

print("모든 이미지 처리 완료!")
