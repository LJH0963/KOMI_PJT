import cv2
import os

# 이미지 폴더 경로 설정
image_folder = 'C:/Users/user/Desktop/img_output/squat/side/for_upload'
video_name = 'C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/data/video/video_side.mp4'
last_output = 'C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/data/video/video_side_h264.mp4'

# 이미지 크기 설정
## 동일 크기이므로 다음과 같이 설정하여 불러옴
frame = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
height, width, layers = frame.shape

# 비디오 저장 시 설정(비디오 이름, 코덱, fps, 영상 크기)
## Trouble shooting : streamlit에서는 H.264 코덱으로 진행해야 mp4 파일 재생 시 문제가 없음
### -> 따라서 이 코덱으로 고정하는 방식으로 변경함.
### -> 
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
os.system(f'ffmpeg -i {video_name} -vcodec libx264 -crf 23 {last_output}')

# 이미지 순서 정렬됨 -> 그 순서로 읽어서 가지고 오도록 함.
for image_file in sorted(os.listdir(image_folder)):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):          # 사실 jpg 하나만 넣어도 되긴 함
        img_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(img_path)
        video.write(frame)

video.release()
print("영상 저장 완료:", video_name)
