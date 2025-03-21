from ultralytics import YOLO
from datetime import datetime
import os
import cv2
import sys
import json
import time

class PoseEstimator(YOLO):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.vcap = None
        self.output_folder = "C:/Users/user/Desktop/img_output/squat/web"

    def start_camera(self, src=0):
        """
        웹캠 초기화 메서드
        """
        self.vcap = cv2.VideoCapture(src)
        if not self.vcap.isOpened():
            raise ConnectionError("❌ 웹캘 연결 실패")
        self.fps = int(self.vcap.get(cv2.CAP_PROP_FPS))

    def real_time_video_recording(self):
        """
        웹캘 실행 후 5초 후보호 2.9초 동안 30fps 기준으로 이미지 먼저 저장 + keypoint JSON 저장
        """
        save_duration_sec = 2.9
        fps = 30
        max_frames = int(save_duration_sec * fps)

        output_img_dir = os.path.join(self.output_folder, 'images')
        output_json_dir = os.path.join(self.output_folder, 'json')
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)

        self.vcap = cv2.VideoCapture(0)
        self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.vcap.set(cv2.CAP_PROP_FPS, fps)

        if not self.vcap.isOpened():
            raise RuntimeError("웹캘을 열 수 없습니다.")

        print("웹캘이 실행되었습니다. 5초 후보호 저장을 시작합니다...")
        start_time = time.time()
        save_started = False
        frame_count = 0
        last_saved_time = None

        while self.vcap.isOpened():
            ret, frame = self.vcap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            current_time = time.time()
            elapsed = current_time - start_time

            if not save_started and elapsed > 5:
                print("저장 시작!")
                save_started = True
                last_saved_time = current_time

            if save_started and frame_count < max_frames:
                if last_saved_time is None or (current_time - last_saved_time) >= 1 / fps:
                    last_saved_time = current_time

                    results = self.predict(frame)
                    keypoints_data = []

                    for result in results:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        scores = result.keypoints.conf.cpu().numpy()
                        person_kps = []

                        for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])):
                            person_kps.append({
                                "id": i,
                                "x": int(kp[0]),
                                "y": int(kp[1]),
                                "confidence": float(score)
                            })
                            if score > 0.5:
                                cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

                        keypoints_data.append({
                            "person_id": 1,
                            "keypoints": person_kps
                        })

                    image_name = f"frame_{frame_count:04d}.jpg"
                    json_name = f"frame_{frame_count:04d}.json"

                    cv2.imwrite(os.path.join(output_img_dir, image_name), frame)

                    result_json = {
                        "status": "success",
                        "pose": keypoints_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    with open(os.path.join(output_json_dir, json_name), 'w', encoding='utf-8') as f:
                        json.dump(result_json, f, indent=4)

                    print(f"{image_name}, {json_name} 저장 완료")
                    frame_count += 1

            if save_started and frame_count >= max_frames:
                print("2.9초 동안 저장 완료. 종료합니다.")
                break

            cv2.imshow("YOLO Pose Estimation", frame)
            if cv2.waitKey(1) == 27:
                break

        self.vcap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pose_estimator = PoseEstimator("yolo11x-pose.pt")
    pose_estimator.real_time_video_recording()
