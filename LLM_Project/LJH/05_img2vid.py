import cv2
import os

# 경로 설정
image_folder = 'C:/Users/user/Desktop/img_output/squat/front'
avi_output = 'C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/data/video/temp_output.avi'
mp4_output = 'C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/data/video/output_video_front.mp4'

# 첫 프레임에서 해상도 얻기
frame = cv2.imread(os.path.join(image_folder, os.listdir(image_folder)[0]))
height, width, layers = frame.shape

# .avi 비디오로 저장 (코덱: XVID)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(avi_output, fourcc, 30, (width, height))

# 이미지 프레임 순차 저장
for image_file in sorted(os.listdir(image_folder)):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(img_path)
        video.write(frame)

video.release()
print("AVI 파일 저장 완료:", avi_output)

# FFmpeg로 H.264 인코딩된 MP4로 변환
print("FFmpeg로 MP4(H.264)로 변환 중...")
ffmpeg_command = f'ffmpeg -y -i "{avi_output}" -vcodec libx264 -crf 23 "{mp4_output}"'
os.system(ffmpeg_command)

# 변환 성공 여부 확인
if os.path.exists(mp4_output) and os.path.getsize(mp4_output) > 1000:
    print("MP4(H.264) 영상 저장 완료:", mp4_output)

    try :
        os.remove(avi_output)
        print("임시 파일 삭제 완료")

    except:
        print("Error : Temp_file is not removed")
else:
    print("MP4 변환 실패 - FFmpeg 설치 또는 입력 확인 필요")
