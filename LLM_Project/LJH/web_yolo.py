import cv2
import tkinter as tk
from tkinter import filedialog
import os
import time
from ultralytics import YOLO
import json
import numpy as np

class PoseRecorder:
    def __init__(self):
        self.cap = None
        self.out = None
        self.video_path = None
        self.image_output_folder = None
        self.json_output_folder = None
        self.mp4_output = None
        self.yolo_model = YOLO("yolo11x-pose.pt")
        self.reference_pose = None
        self.mask = None
        self.output_folder = self.select_output_folder()
        self.setup_folders()
        self.load_reference_data()

    def select_output_folder(self):             ## 결과물을 저장할 폴더를 선택
        root = tk.Tk()
        root.withdraw()
        output_folder = filedialog.askdirectory(title='결과 저장 폴더를 선택하세요')
        if not output_folder:
            print("결과를 저장할 폴더를 선택하지 않았습니다. 프로그램을 종료합니다.")
            exit()
        return output_folder

    def setup_folders(self):                    ## 각 저장될 파일 및 폴더 생성
        self.video_path = os.path.join(self.output_folder, 'recorded_video.avi')
        self.image_output_folder = os.path.join(self.output_folder, 'image')
        self.json_output_folder = os.path.join(self.output_folder, 'json')
        self.mp4_output = os.path.join(self.output_folder, 'user_output_front.mp4')
        os.makedirs(self.image_output_folder, exist_ok=True)
        os.makedirs(self.json_output_folder, exist_ok=True)

    def load_reference_data(self):              ## Reference로 쓰일 마스크 이미지와 Json 불러오기
        mask_path = 'C:/Users/user/Desktop/data/frame100_mask_rgba.png'
        pose_path = "C:/Users/user/Desktop/img_output/squat/front_json/json/use/frame_000.json"
        self.mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if self.mask is None:
            print("마스크 이미지를 찾을 수 없습니다.")
            exit()
        self.reference_pose = self.load_reference_pose(pose_path)

    def pose_similar(self, current_pose, threshold_px=20, ratio=0.7):               ## reference와의 유사도
        match_count = total_count = 0
        for ref, cur in zip(self.reference_pose, current_pose):
            if ref[0] is not None and cur[0] is not None:
                if np.linalg.norm(np.array(ref) - np.array(cur)) <= threshold_px:
                    match_count += 1
                total_count += 1
        return (match_count / total_count) >= ratio if total_count > 0 else False

    def overlay_mask(self, frame, alpha_value=100):                                ## 웹캠 화면에 mask overlay
        mask_resized = cv2.resize(self.mask, (frame.shape[1], frame.shape[0]))
        mask_rgb = mask_resized[:, :, :3].astype(np.uint8)
        mask_alpha = mask_resized[:, :, 3].astype(np.uint8)
        object_mask = (mask_alpha > 0).astype(np.uint8)
        custom_alpha = np.where(object_mask == 1, alpha_value, 0).astype(np.uint8)
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        for c in range(3):
            frame_rgba[:, :, c] = (
                frame_rgba[:, :, c] * (1 - custom_alpha / 255.0) +
                mask_rgb[:, :, c] * (custom_alpha / 255.0)
            ).astype(np.uint8)
        return cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

    def wait_for_alignment(self):                                   ## 정렬 대기용
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("초기 정렬 대기 중...")
        while True:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            results = self.yolo_model.predict(source=frame, stream=False, verbose=False)
            keypoints = None
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()[0]
                    break
            vis_frame = self.overlay_mask(frame.copy())
            if keypoints is not None and self.pose_similar(keypoints):
                cv2.putText(vis_frame, "Pose Aligned! Starting soon...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.imshow("Pose Alignment", vis_frame)
                cv2.waitKey(1000)
                break
            else:
                cv2.putText(vis_frame, "Align your pose with the reference", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Pose Alignment", vis_frame)
            if cv2.waitKey(1) == 27:
                self.cap.release()
                cv2.destroyAllWindows()
                exit()

    def record_video(self, duration=2.9, countdown=5):                          ### 카운트 다운 후 비디오 녹화
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (1280, 720))
        print("정렬 완료! 5초 후 녹화가 시작됩니다.")
        start = time.time()
        while time.time() - start < countdown:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, str(countdown - int(time.time() - start)), (600, 380), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
            cv2.imshow('Recording Video', frame)
            if cv2.waitKey(1) == 27:
                exit()
        start_record = time.time()
        while time.time() - start_record <= duration:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            self.out.write(frame)
            cv2.imshow('Recording Video', frame)
            if cv2.waitKey(1) == 27:
                break
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def extract_frames(self, max_frames=87):                        ## 녹화된 영상 30fps / 87프레임 추출
        cap = cv2.VideoCapture(self.video_path)
        for idx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            output_image_path = os.path.join(self.image_output_folder, f"frame_{idx:03d}.jpg")
            cv2.imwrite(output_image_path, frame)
        cap.release()

    def run_pose_estimation(self):                              ## 추출된 이미지로 estimation
        COCO_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                          "left_wrist", "right_wrist", "left_hip", "right_hip",
                          "left_knee", "right_knee", "left_ankle", "right_ankle"]
        for img_file in sorted(os.listdir(self.image_output_folder)):
            if not img_file.endswith('.jpg'):
                continue
            img_path = os.path.join(self.image_output_folder, img_file)
            image = cv2.imread(img_path)
            results = self.yolo_model(image)
            json_data = {'image_name': img_file, 'bboxes': [], 'keypoints': []}
            keypoints_dict = {part: {"x": None, "y": None, "confidence": 0.0} for part in COCO_KEYPOINTS}
            for result in results:
                if result.boxes is not None:
                    for bbox, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                        json_data['bboxes'].append({
                            'class': int(cls), 'bbox': list(map(int, bbox)), 'confidence': float(conf)
                        })
                if result.keypoints is not None:
                    for idx, (kp, score) in enumerate(zip(result.keypoints.xy.cpu().numpy()[0], result.keypoints.conf.cpu().numpy()[0])):
                        x, y, conf = int(kp[0]), int(kp[1]), float(score)
                        keypoints_dict[COCO_KEYPOINTS[idx]] = {
                            'x': x if conf > 0.1 else None,
                            'y': y if conf > 0.1 else None,
                            'confidence': conf
                        }
            json_data["keypoints"] = [
                {"part": part, **keypoints_dict[part]} for part in COCO_KEYPOINTS
            ]
            with open(os.path.join(self.json_output_folder, img_file.replace('.jpg', '.json')), 'w', encoding='utf-8') as jf:
                json.dump(json_data, jf, indent=4)

    def convert_to_mp4(self):                           ## 녹화된 영상 mp4 등으로 코덱 변경
        os.system(f'ffmpeg -y -i "{self.video_path}" -vcodec libx264 -crf 23 "{self.mp4_output}"')
        if os.path.exists(self.mp4_output) and os.path.getsize(self.mp4_output) > 1000:
            print("MP4 저장 완료:", self.mp4_output)
            try:
                os.remove(self.video_path)
            except:
                print("임시 파일 삭제 실패")
        else:
            print("MP4 변환 실패 - FFmpeg 설치 필요")

    def run(self):
        self.wait_for_alignment()
        self.record_video()
        self.extract_frames()
        self.run_pose_estimation()
        self.convert_to_mp4()

if __name__ == '__main__':
    PoseRecorder().run()
