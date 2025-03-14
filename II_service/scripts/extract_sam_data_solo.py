import cv2
import numpy as np
import torch
import os
import glob
import pandas as pd
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# 📌 입력 및 출력 디렉토리 설정
input_dir = "data/solo_dance"  # 입력 이미지 폴더
output_dir = "data/solo_dance_sam"  # 세그멘테이션 결과 이미지 저장 폴더
csv_output_dir = "data/solo_dance_csv"  # CSV 저장 폴더

os.makedirs(output_dir, exist_ok=True)  # 출력 이미지 폴더 생성
os.makedirs(csv_output_dir, exist_ok=True)  # CSV 저장 폴더 생성

# 📌 디바이스 설정 (GPU 사용 가능 여부)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

# 📌 SAM 모델 로드
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 📌 입력 폴더 내 모든 jpg 파일 찾기
image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

# 📌 파일이 없을 경우 경고
if not image_paths:
    print("⚠ 경고: 해당 디렉토리에 JPG 파일이 없습니다!")

# 📌 모든 이미지에 대해 반복 처리
# for image_path in image_paths[:3]:  # 예제: 처음 3개 이미지 처리
for image_path in image_paths:
    # 원본 이미지 파일 이름 가져오기
    image_name = os.path.basename(image_path).split(".")[0]  # 확장자 제거
    output_path = os.path.join(output_dir, f"{image_name}_mask.jpg")  # 마스크 저장 경로
    csv_output_path = os.path.join(csv_output_dir, f"{image_name}.csv")  # CSV 저장 경로

    print(f"▶ 처리 중: {image_path} → {output_path}, {csv_output_path}")

    # 📌 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 오류: {image_path} 를 불러올 수 없습니다.")
        continue
    image = cv2.resize(image, (640, 480))  # resize 진행

    # 📌 객체 검출 수행
    results = model.predict(source=image, conf=0.6)

    # 📌 검출된 경계 상자 추출
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    
    # YOLO가 감지한 박스가 없으면 건너뜀
    if len(bboxes) == 0:
        print(f"❌ 객체 감지 실패: {image_name}, 건너뜁니다.")
        continue

    # 📌 이미지 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 📌 SAM에 이미지 설정
    predictor.set_image(image_rgb)

    # 📌 경계 상자를 사용하여 세그멘테이션 마스크 생성
    transformed_boxes = predictor.transform.apply_boxes_torch(
        torch.tensor(bboxes, dtype=torch.float32), image_rgb.shape[:2]
    )
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # 📌 결과 저장을 위한 빈 마스크 생성 (배경: 0, 객체: 1)
    segmentation_result = np.zeros_like(image_rgb[:, :, 0], dtype=np.uint8)  # 전체 0 (배경)

    # 📌 마스크 적용 (객체 부분을 1로 변경)
    for mask in masks:
        mask = mask.cpu().numpy().astype(np.uint8).squeeze()  # 차원 축소 후 적용
        segmentation_result[mask > 0] = 1  # 객체 부분을 1로 설정 (배경은 0)

    # 📌 결과 저장 (마스크 이미지)
    cv2.imwrite(output_path, segmentation_result * 255)  # 흑백 이미지로 저장
    print(f"✅ 마스크 이미지 저장 완료: {output_path}")

    # 📌 2D 배열을 1D 벡터로 변환 후 CSV 저장
    flat_array = segmentation_result.flatten()  # 2D → 1D 배열 변환
    df = pd.DataFrame([flat_array])  # 단일 행 데이터프레임 생성
    df.to_csv(csv_output_path, index=False, header=False)  # CSV 저장

    print(f"✅ CSV 저장 완료: {csv_output_path}")

print("🎉 모든 이미지 처리 완료!")
