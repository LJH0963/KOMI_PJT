import cv2
import os

# 이미지가 저장된 폴더 경로
image_folder = 'C:/Users/user/Desktop/img_output/squat/front'
video_name = 'C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/data/video/output_video.mp4'

# 이미지 크기 설정 (이미지들이 동일한 크기여야 함)
frame = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
height, width, layers = frame.shape

# 비디오 저장 설정 (코덱, FPS, 해상도)
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

# 이미지들을 순서대로 읽어서 영상으로 저장
for image_file in sorted(os.listdir(image_folder)):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(img_path)
        video.write(frame)

video.release()
print("영상 저장 완료:", video_name)
