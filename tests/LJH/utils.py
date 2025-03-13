from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import torch
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class PoseEstimator:
    def __init__(self, model_path):
        """
        YOLO 모델을 불러와 초기화.
        """
        self.model = YOLO(model_path)

    def detect_image_pose(self, frame):
        """
        주어진 프레임에서 포즈 감지 수행 후, 키포인트 좌표 반환 함수
        """
        results = self.model(frame)
        pose_data = []

        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()       # 키포인트 (x, y) 좌표 추출
            scores = result.keypoints.conf.cpu().numpy()        # 신뢰도 값 추출

            keypoints_list = []
            for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])):  
                if score > 0.5:                                 # 신뢰도 50% 이상인 경우만 포함
                    keypoints_list.append({
                        "id": i,                                # 관절 ID
                        "x": int(kp[0]),                        # x 좌표
                        "y": int(kp[1]),                        # y 좌표
                        "confidence": float(score)              # 신뢰도 값
                    })
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)  # 키포인트 시각화

            # 감지된 사람 데이터 저장
            pose_data.append({
                "person_id": 1,                                 # 현재는 한 명 감지 기준
                "keypoints": keypoints_list
            })

        return pose_data, frame
    

    def detect_video_pose(self, frame):
        """
        실시간 웹캠을 사용하여 좌표 감지 및 화면 출력
        """
        vcap = cv2.VideoCapture(0)                              # 기본 웹캠 사용

        # 웹캠 정상 작동 확인
        if not vcap.isOpened():
            print("웹캠을 열 수 없습니다.")
            sys.exit()

        # 웹캠 실시간 처리 루프
        while vcap.isOpened():
            # 1. 프레임 읽기기
            ret, frame = vcap.read()            # ret: 작동 여부, # frame: 카메라로 받은 이미지

            # 2. 웹캠 오류 체크
            if not ret:
                print("웹캠 프레임을 가져올 수 없습니다.")
                break
            
            # 3. 좌우 반전 (True)
            frame = cv2.flip(frame, 1)

            # 4. 모델 활용하여 이미지 감지
            model = PoseEstimator('./tests/KHS/models/yolov8n-pose.pt')
            results = model(frame)

            # 5. 감지된 좌표 값 화면에 표시
            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy()       # tensor -> numpy 배열로 변경할때는 .cpu().numpy() 사용
                scores = result.keypoints.conf.cpu().numpy()        # tensor -> numpy 배열로 변경할때는 .cpu().numpy() 사용
                print(keypoints.shape, scores.shape)                # keypoints.shape = [사람 수, 각 사람의 키포인트 수(관절), 각 키포인트(관절)의 좌표]  

                # Keypoints 데이터 번호 정리
                """
                # 각 키포인트(관절)의 좌표 순번 : COCO Dataset 기준
                # 0	Nose(코)                        얼굴 중심점
                # 1	Left Eye(왼쪽 눈)               얼굴 왼쪽 위치
                # 2	Right Eye (오른쪽 눈)	        얼굴 오른쪽 위치
                # 3	Left Ear (왼쪽 귀)	            얼굴 왼쪽 끝
                # 4	Right Ear (오른쪽 귀)	        얼굴 오른쪽 끝
                # 5	Left Shoulder (왼쪽 어깨)	    상체 위치
                # 6	Right Shoulder (오른쪽 어깨)	상체 위치
                # 7	Left Elbow (왼쪽 팔꿈치)	    팔 관절
                # 8	Right Elbow (오른쪽 팔꿈치)	    팔 관절
                # 9	Left Wrist (왼쪽 손목)	        손 위치
                # 10 Right Wrist (오른쪽 손목)	    손 위치
                # 11 Left Hip (왼쪽 엉덩이)	        하체 위치
                # 12 Right Hip (오른쪽 엉덩이)	    하체 위치
                # 13 Left Knee (왼쪽 무릎)	        다리 관절
                # 14 Right Knee (오른쪽 무릎)	    다리 관절
                # 15 Left Ankle (왼쪽 발목)	        발 위치
                # 16 Right Ankle (오른쪽 발목)	    발 위치
                """
                
                # 저장소 만들기
                pose_data = []
                keypoints_list = []

                for i, (kp, score) in enumerate(zip(keypoints[0], scores[0])):
                    # 신뢰도 50% 이상인 경우만 포함   
                    if score > 0.5:
                        keypoints_list.append({
                            "id": i,                    # 관절 ID (COCO 데이터셋 기준)
                            "x": int(kp[0]),            # x 좌표
                            "y": int(kp[1]),            # y 좌표
                            "confidence": float(score)  # 신뢰도 값 (0~1)
                        })

                        # Keypoints 시각화
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

                # 사람 단위로 데이터 저장
                pose_data.append({
                    "person_id": 1,                     # 감지된 사람 ID
                    "keypoints": keypoints_list         # 관절 리스트
                })

            # 최종 데이터 구조 (JSON 형태로 저장 : FastAPI Data Default)
            pose_response = {
                "status": "success",  # 요청 성공 여부
                "pose": pose_data,  # 감지된 포즈 데이터
                "timestamp": datetime.utcnow().isoformat()  # 타임스탬프
            }

            print(pose_response)


            # 감지된 결과 화면 출력
            cv2.imshow("YOLO Pose Estimation", frame)

            # 꺼지는 조건 설정
            key = cv2.waitKey(1)

            # ESC : 27 (아스키코드)
            if key == 27:
                break

        # 웹캠 종료 및 창 닫기
        vcap.release()
        cv2.destroyAllWindows()