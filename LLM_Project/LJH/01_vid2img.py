import cv2
import os
import tkinter as tk
from tkinter import filedialog

def select_video_file():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    return file_path

def select_save_folder():
    folder_path = filedialog.askdirectory(title="Select Save Folder")
    return folder_path

# Tkinter GUI 숨기기
root = tk.Tk()
root.withdraw()

# 비디오 파일과 저장 폴더 선택
video_filepath = select_video_file()
save_folder = select_save_folder()

if not video_filepath or not save_folder:
    print("No file or folder selected.")
    exit(0)

video = cv2.VideoCapture(video_filepath)

if not video.isOpened():
    print("Could not Open :", video_filepath)
    exit(0)

# 비디오 정보 출력
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)

# 저장할 디렉토리 설정
video_name = os.path.splitext(os.path.basename(video_filepath))[0]
save_path = os.path.join(save_folder, video_name)

try:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
except OSError:
    print('Error: Creating directory.', save_path)
    exit(0)

count = 0

while video.isOpened():
    ret, image = video.read()
    if not ret:
        break
    
    if int(video.get(30)) % fps == 0:  # 1초마다 프레임 저장
        frame_filename = os.path.join(save_path, f"frame_{count:03d}.jpg")
        cv2.imwrite(frame_filename, image)
        print('Saved frame number :', int(video.get(1)))
        count += 1

video.release()
print("Frame extraction completed.")
