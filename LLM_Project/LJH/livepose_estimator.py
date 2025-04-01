from ultralytics import YOLO
from datetime import datetime
import cv2
import json
import base64

# Image 및 Detecting한 Pose 데이터를 하나의 딕셔너리로 정리하여 만드는 함수
def prepare_pose_data(pose_data, frame=None, include_image=True):
    result = {
        "status": "success",
        "pose": pose_data,
        "timestamp": datetime.utcnow().isoformat()
    }

    if include_image and frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        result["image"] = base64.b64encode(buffer).decode('utf-8')

    return result



class LivePoseEstimator(YOLO):
    # Yolo Pose 모델 불러오기
    def __init__(self, model_path):
        super().__init__(model_path)
        self.vcap = None
        self.all_pose_data = []

    # 웹캠 실행 -> 해상도 및 FPS 설정하기
    def start_camera(self, src=0, width=1280, height=720, fps=30):
        self.vcap = cv2.VideoCapture(src)
        if not self.vcap.isOpened():
            raise ConnectionError("웹캠 연결 실패")

        self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.vcap.set(cv2.CAP_PROP_FPS, fps)
        print(f"카메라 시작: {width}x{height} @ {fps}fps")

    # 실시간 Detecting
    def real_time_video_detecting(self, save_json_path="C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/webcam_output_json/saved_pose_data.json"):
        if self.vcap is None:
            raise ValueError("카메라가 초기화되지 않았습니다.")

        while self.vcap.isOpened():
            ret, frame = self.vcap.read()
            if not ret:
                print("프레임 수신 실패")
                break

            frame = cv2.flip(frame, 1)
            results = self.predict(frame)

            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy()
                scores = result.keypoints.conf.cpu().numpy()

                pose_data = []
                keypoints_list = []

                for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])):
                    if score > 0.5:
                        keypoints_list.append({
                            "id": i,
                            "x": int(kp[0]),
                            "y": int(kp[1]),
                            "confidence": float(score)
                        })
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

                pose_data.append({
                    "person_id": 1,
                    "keypoints": keypoints_list
                })

                pose_response = prepare_pose_data(pose_data, frame, include_image=True)
                self.all_pose_data.append(pose_response)

                print(pose_response["timestamp"], "- Keypoints 수:", len(keypoints_list))

            cv2.imshow("YOLO Pose Estimation", frame)
            if cv2.waitKey(1) == 27:
                break

        self.vcap.release()
        cv2.destroyAllWindows()

        # 카메라 종료 시 Json 데이터 전체 저장
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_pose_data, f, indent=2)
        print(f"✅ 전체 포즈 데이터 저장 완료: {save_json_path}")

# 코드 오류 확인용 실행
if __name__ == "__main__":
    model_path = "yolo11x-pose.pt"  # YOLO-Pose 설정
    estimator = LivePoseEstimator(model_path)
    estimator.start_camera(src=0, width=1280, height=720, fps=30)
    estimator.real_time_video_detecting()

### 실시간으로 디텍팅 되는 것으로 확인. -> Json으로 떨궈지는 것도 확인