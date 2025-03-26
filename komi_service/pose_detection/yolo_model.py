# YOLO11 포즈 감지 모델 파일

# 이 파일은 YOLO11 기반의 포즈 감지 모델을 구현하기 위한 파일입니다. 

import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image

# COCO Keypoint 이름 리스트
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO 데이터셋 기준의 관절 연결 정보
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # 팔 (오른쪽, 왼쪽)
    (11, 13), (13, 15), (12, 14), (14, 16),  # 다리 (오른쪽, 왼쪽)
    (5, 6), (11, 12), (5, 11), (6, 12)  # 몸통 연결
]

class YoloPoseModel:
    """
    YOLO11 기반 포즈 감지 모델 클래스
    """
    def __init__(self, model_path="yolo11x-pose.pt"):
        """
        모델 초기화
        
        Args:
            model_path (str): YOLO 모델 경로
        """
        try:
            self.model = YOLO(model_path)
            self.is_loaded = True
            print(f"YOLO 모델 로드 성공: {model_path}")
        except Exception as e:
            self.is_loaded = False
            print(f"YOLO 모델 로드 실패: {str(e)}")

    def detect_pose(self, image, conf_threshold=0.5):
        """
        이미지에서 포즈 감지 수행
        
        Args:
            image: NumPy 배열 또는 이미지 경로
            conf_threshold: 신뢰도 임계값
            
        Returns:
            results_dict: 감지 결과가 포함된 딕셔너리
        """
        if not self.is_loaded:
            return {"error": "모델이 로드되지 않았습니다"}
        
        # 이미지 처리
        if isinstance(image, str):
            if image.startswith("data:image"):
                # Base64 인코딩 이미지 처리
                try:
                    encoded_data = image.split(',')[1]
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as e:
                    return {"error": f"Base64 이미지 디코딩 실패: {str(e)}"}
            else:
                # 파일 경로
                try:
                    image = cv2.imread(image)
                except Exception as e:
                    return {"error": f"이미지 로드 실패: {str(e)}"}
        
        if image is None:
            return {"error": "유효하지 않은 이미지"}
            
        # YOLO 모델로 포즈 감지
        results = self.model(image)
        
        # 결과 처리
        results_dict = {
            "bboxes": [],
            "keypoints": []
        }
        
        for result in results:
            # 바운딩 박스 정보 추출
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    if conf > conf_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        results_dict["bboxes"].append({
                            "class": int(cls),
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(conf)
                        })
            
            # 키포인트 정보 추출
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                
                for person_idx, (kps, confs) in enumerate(zip(keypoints, confidences)):
                    keypoints_data = []
                    
                    for i, (kp, conf) in enumerate(zip(kps, confs)):
                        x, y = int(kp[0]), int(kp[1])
                        keypoints_data.append({
                            "part": COCO_KEYPOINTS[i],
                            "x": x if conf > 0.1 else None,
                            "y": y if conf > 0.1 else None,
                            "confidence": float(conf)
                        })
                    
                    results_dict["keypoints"].append(keypoints_data)
        
        return results_dict
    
    def draw_pose(self, image, pose_data, draw_skeleton=True, draw_keypoints=True):
        """
        감지된 포즈를 이미지에 시각화
        
        Args:
            image: NumPy 배열 이미지
            pose_data: detect_pose() 함수의 결과
            draw_skeleton: 스켈레톤 그리기 여부
            draw_keypoints: 키포인트 그리기 여부
            
        Returns:
            시각화된 이미지
        """
        img_copy = image.copy()
        
        # 바운딩 박스 그리기
        for bbox in pose_data.get("bboxes", []):
            x1, y1, x2, y2 = bbox["bbox"]
            conf = bbox["confidence"]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_copy, f"conf: {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 키포인트와 스켈레톤 그리기
        for person_keypoints in pose_data.get("keypoints", []):
            # 키포인트 딕셔너리로 변환
            kp_dict = {kp["part"]: kp for kp in person_keypoints}
            
            # 키포인트 그리기
            if draw_keypoints:
                for kp in person_keypoints:
                    if kp["x"] is not None and kp["y"] is not None and kp["confidence"] > 0.5:
                        cv2.circle(img_copy, (kp["x"], kp["y"]), 5, (0, 0, 255), -1)
            
            # 스켈레톤 그리기
            if draw_skeleton:
                for joint1_idx, joint2_idx in SKELETON:
                    joint1_name = COCO_KEYPOINTS[joint1_idx]
                    joint2_name = COCO_KEYPOINTS[joint2_idx]
                    
                    joint1 = kp_dict.get(joint1_name, {})
                    joint2 = kp_dict.get(joint2_name, {})
                    
                    if (joint1.get("x") is not None and joint1.get("y") is not None and 
                        joint2.get("x") is not None and joint2.get("y") is not None and
                        joint1.get("confidence", 0) > 0.5 and joint2.get("confidence", 0) > 0.5):
                        cv2.line(img_copy, 
                                (joint1["x"], joint1["y"]), 
                                (joint2["x"], joint2["y"]), 
                                (0, 255, 0), 2)
        
        return img_copy

    def process_base64_image(self, base64_image):
        """
        Base64 인코딩된 이미지 처리
        
        Args:
            base64_image: Base64 인코딩된 이미지 문자열
            
        Returns:
            NumPy 배열 이미지
        """
        try:
            # 헤더 제거
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            # 디코딩 후 이미지로 변환
            image_bytes = base64.b64decode(base64_image)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            return image
        except Exception as e:
            print(f"Base64 이미지 처리 오류: {str(e)}")
            return None
    
    def image_to_base64(self, image):
        """
        NumPy 배열 이미지를 Base64 문자열로 변환
        
        Args:
            image: NumPy 배열 이미지
            
        Returns:
            Base64 인코딩된 이미지 문자열
        """
        try:
            # OpenCV BGR 이미지를 RGB로 변환
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # PIL 이미지로 변환 후 JPEG 형식으로 인코딩
            pil_image = Image.fromarray(image_rgb)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            
            # Base64 인코딩
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_image}"
        except Exception as e:
            print(f"이미지를 Base64로 변환 오류: {str(e)}")
            return None 