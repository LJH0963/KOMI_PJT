from fastapi import FastAPI, WebSocket, HTTPException, Body, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
import threading
from contextlib import asynccontextmanager
import mimetypes
import shutil
# LLM 관련 라이브러리 추가
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import re
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

# MIME 타입 등록
mimetypes.add_type("video/mp4", ".mp4")

# 영상 저장 경로 설정
VIDEO_STORAGE_PATH = os.environ.get("VIDEO_STORAGE_PATH", "./video_uploads")
VIDEO_ANALYSIS_PATH = os.environ.get("VIDEO_ANALYSIS_PATH", "./video_analysis")

# YOLO 모델 경로
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolo11x-pose.pt")

# 참조 포즈 경로 설정
REFERENCE_POSES_PATH = os.environ.get("REFERENCE_POSES_PATH", "./data")

# 데이터 디렉토리 설정
DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY", "./data")

# 저장소: 카메라 ID -> 이미지 데이터
latest_image_data: Dict[str, str] = {}
latest_timestamps: Dict[str, datetime] = {}

# 활성 연결
active_connections: Set[WebSocket] = set()
camera_info: Dict[str, dict] = {}

# 락
data_lock = threading.Lock()

# 상태 관리
app_state = {
    "is_running": True,
    "connected_cameras": 0,
    "active_websockets": 0,
    "start_time": datetime.now(),
    "last_connection_cleanup": datetime.now()
}

# 운동 관련 데이터
exercise_data = {
    "exercises": [
        {
            "id": "squat",
            "name": "스쿼트",
            "description": "기본 하체 운동",
            "guide_videos": {
                "front": "/squat/front.mp4",
                "side": "/squat/side.mp4"
            }
        },
        {
            "id": "pushup",
            "name": "푸시업",
            "description": "상체 근력 운동",
            "guide_videos": {
                "front": "/pushup/front.mp4",
                "side": "/pushup/side.mp4"
            }
        },
        {
            "id": "lunge",
            "name": "런지",
            "description": "하체 균형 운동",
            "guide_videos": {
                "front": "/lunge/front.mp4",
                "side": "/lunge/side.mp4"
            }
        }
    ]
}

# ----- LLM 모델 관련 함수 추가 -----

# LLM 설정 함수
def load_llm():
    """OpenAI ChatGPT 자동 설정 함수"""
    return ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini" 
    )

def format_docs(docs):
    """문서들을 하나의문자열로 포맷팅하는 함수"""
    return "\n\n".join(doc.page_content for doc in docs)

def format_response(text: str) -> str:
    """모든 출력 텍스트를 가독성 좋게 정리하는 함수"""
    # 마침표, 물음표, 느낌표 뒤에 줄바꿈 추가
    lines = re.sub(r'([.!?])\s+', r'\1\n', text.strip())
    return lines

# 벡터스토어 설정
def create_vectorstore(pdf_docs_dir="../chroma_db/pdf_docs"):
    """벡터스토어 생성 또는 로드"""
    # 디렉토리가 없으면 생성
    os.makedirs(pdf_docs_dir, exist_ok=True)
    
    return Chroma(
        persist_directory=pdf_docs_dir,
        embedding_function=OpenAIEmbeddings()
    )

# 자연어 프롬프트 생성 함수
def generate_summary_prompt(input_data: dict) -> str:
    """분석 결과로부터 LLM 프롬프트 생성"""
    # 문제가 있는 관절 카운트
    joint_counter = Counter()
    
    # 프레임별 분석 결과가 있으면 순회
    if "frame_results" in input_data:
        for frame_name, frame_data in input_data["frame_results"].items():
            # 각도 차이가 15도 이상인 관절 찾기
            failed_joints = []
            for key, value in frame_data.items():
                if key.endswith('_angle_diff') and value is not None and abs(value) >= 15:
                    failed_joints.append(key)
                    joint_counter[key] += 1
    
    # 관절명 매핑
    part_names = {
        # Front view
        "left_hip_angle_diff": "왼쪽 고관절",
        "right_hip_angle_diff": "오른쪽 고관절",
        "left_knee_angle_diff": "왼쪽 무릎",
        "right_knee_angle_diff": "오른쪽 무릎",

        # Side view
        "left_shoulder_angle_diff": "왼쪽 어깨",
        "left_hip_angle_diff": "왼쪽 고관절",
        "left_knee_angle_diff": "왼쪽 무릎",
    }

    # 관절별 잘못된 횟수를 텍스트로 나열
    front_lines = []
    side_lines = []

    for joint, count in joint_counter.items():
        name = part_names.get(joint, joint)
        line = f"- {name}: {count}회"
        if "angle" in joint and ("left" in joint or "right" in joint):
            front_lines.append(line)
        else:
            side_lines.append(line)

    # 최종 LLM용 프롬프트 생성
    prompt = "운동 영상에서 다음 부위에 자주 문제가 발생했습니다:\n"

    if front_lines:
        prompt += "\n[정면 View 기준 문제 부위]\n" + "\n".join(front_lines)
    if side_lines:
        prompt += "\n\n[측면 View 기준 문제 부위]\n" + "\n".join(side_lines)

    prompt += (
        "\n\n이러한 문제가 왜 발생할 수 있는지, 그리고 어떻게 개선하면 좋을지 "
        "운동 전문가 입장에서 설명해 주세요."
    )
    return prompt

# LLM을 사용한 분석 요약 생성 함수
async def generate_llm_analysis(pose_eval_path: str, output_json_path: str):
    """LLM을 사용하여 포즈 분석 결과를 해석하고 개선점을 제안"""
    try:
        # 1. 포즈 평가 결과 로드
        with open(pose_eval_path, 'r', encoding='utf-8') as f:
            pose_data = json.load(f)
        
        # 2. 자연어 프롬프트 생성
        question = generate_summary_prompt(pose_data)
        
        # 3. LLM 모델 설정
        llm = load_llm()
        
        # 4. 벡터스토어 설정 및 검색기 생성
        vectorstore = create_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 5. 프롬프트 템플릿 설정
        prompt_template = PromptTemplate.from_template(
            "{context}\n\n{question}")
        
        # 6. Langchain 설정
        rag_chain = (
            RunnableMap({
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
                })
            | prompt_template
            | llm
            | StrOutputParser()
            | RunnableLambda(format_response)
        )
        
        # 7. LLM 분석 실행
        llm_response = rag_chain.invoke(question)
        
        # 8. 결과 저장
        result = {
            "prompt": question,
            "llm_analysis": llm_response,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        return result
    
    except Exception as e:
        error_result = {
            "error": str(e),
            "message": "LLM 분석 중 오류가 발생했습니다",
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=4, ensure_ascii=False)
        
        return error_result

# ----- 포즈 평가를 위한 유틸리티 함수 -----

# JSON에서 keypoints 불러오기
def load_keypoints_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("keypoints", [])

# 특정 관절 좌표 반환
def get_point(kps, part_name):
    for kp in kps:
        if kp['part'] == part_name and kp['x'] is not None and kp['y'] is not None:
            return (kp['x'], kp['y'])
    return None

# 관절 3점으로 각도 계산
def compute_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# 각도 유사도 평가
def cosine_similarity(vec1, vec2):
    if None in vec1 or None in vec2:
        return 0.0
    vec1 = [v if v is not None else 0 for v in vec1]
    vec2 = [v if v is not None else 0 for v in vec2]
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# 포즈 각도 평가 함수
async def evaluate_pose_angles(target_json_dir, exercise_id, output_json_path):
    """
    포즈 각도를 평가하고 결과를 JSON으로 저장
    Args:
        target_json_dir: 분석할 JSON 파일 디렉토리
        exercise_id: 운동 ID (squat, pushup, lunge 등)
        output_json_path: 결과를 저장할 JSON 파일 경로
    Returns:
        평가 결과 요약
    """
    # 운동 유형에 맞는 참조 포즈 디렉토리 결정 (front_json 우선, 없으면 side_json)
    front_ref_dir = os.path.join(REFERENCE_POSES_PATH, exercise_id, "front_json")
    side_ref_dir = os.path.join(REFERENCE_POSES_PATH, exercise_id, "side_json")
    
    # 우선순위에 따라 참조 포즈 디렉토리 선택
    if os.path.exists(front_ref_dir) and os.listdir(front_ref_dir):
        ref_pose_dir = front_ref_dir
        view_type = "front"
    elif os.path.exists(side_ref_dir) and os.listdir(side_ref_dir):
        ref_pose_dir = side_ref_dir
        view_type = "side"
    else:
        # 기본값으로 squat의 front_json 사용
        default_front_dir = os.path.join(REFERENCE_POSES_PATH, "squat", "front_json")
        default_side_dir = os.path.join(REFERENCE_POSES_PATH, "squat", "side_json")
        
        if os.path.exists(default_front_dir) and os.listdir(default_front_dir):
            ref_pose_dir = default_front_dir
            view_type = "front"
        elif os.path.exists(default_side_dir) and os.listdir(default_side_dir):
            ref_pose_dir = default_side_dir
            view_type = "side"
        else:
            print(f"참조 포즈를 찾을 수 없습니다. exercise_id: {exercise_id}")
            return {"status": "error", "message": "참조 포즈를 찾을 수 없습니다", "exercise_id": exercise_id}
    
    print(f"사용할 참조 포즈 디렉토리: {ref_pose_dir}, 시점: {view_type}")
    
    # 결과 저장 딕셔너리
    result_dict = {}
    
    # 타겟 JSON 파일 목록
    target_files = sorted([f for f in os.listdir(target_json_dir) if f.endswith('.json')])
    if not target_files:
        return {"status": "error", "message": "분석할 JSON 파일이 없습니다"}
    
    # 참조 JSON 파일 목록
    ref_files = sorted([f for f in os.listdir(ref_pose_dir) if f.endswith('.json')])
    if not ref_files:
        return {"status": "error", "message": "참조 포즈 JSON 파일이 없습니다", "ref_dir": ref_pose_dir}
    
    # 참조 데이터 선택 - 첫 번째 파일 사용
    ref_json_path = os.path.join(ref_pose_dir, ref_files[0])
    ref_keypoints = load_keypoints_from_json(ref_json_path)
    
    # 각 타겟 파일 평가
    for target_file in target_files:
        target_json_path = os.path.join(target_json_dir, target_file)
        target_keypoints = load_keypoints_from_json(target_json_path)
        
        # 각도 계산 및 유사도 평가
        result_dict[target_file] = compute_pose_result(ref_keypoints, target_keypoints)
    
    # 평균 점수 계산
    scores = [result.get("cosine_similarity", 0) for result in result_dict.values()]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # 전체 결과 요약
    summary = {
        "exercise_id": exercise_id,
        "view_type": view_type,
        "reference_dir": ref_pose_dir,
        "frames_analyzed": len(result_dict),
        "average_similarity": round(avg_score, 4),
        "timestamp": datetime.now().isoformat()
    }
    
    # 결과를 JSON 파일로 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "frame_results": result_dict
        }, f, indent=4, ensure_ascii=False)
    
    return summary

def compute_pose_result(kps1, kps2):
    """
    두 포즈 간의 각도 차이 및 유사도 계산
    Args:
        kps1: 참조 포즈 키포인트
        kps2: 평가 대상 포즈 키포인트
    Returns:
        각도 차이 및 유사도 결과
    """
    def safe_diff(ref, eval):
        return abs(ref - eval) if ref is not None and eval is not None else 999

    def pass_check(diff):
        return 1 if diff <= 15 else 0

    # 고관절
    lh_ref = compute_angle(get_point(kps1, 'left_shoulder'), get_point(kps1, 'left_hip'), get_point(kps1, 'left_knee'))
    rh_ref = compute_angle(get_point(kps1, 'right_shoulder'), get_point(kps1, 'right_hip'), get_point(kps1, 'right_knee'))
    lh_eval = compute_angle(get_point(kps2, 'left_shoulder'), get_point(kps2, 'left_hip'), get_point(kps2, 'left_knee'))
    rh_eval = compute_angle(get_point(kps2, 'right_shoulder'), get_point(kps2, 'right_hip'), get_point(kps2, 'right_knee'))

    # 무릎
    lk_ref = compute_angle(get_point(kps1, 'left_hip'), get_point(kps1, 'left_knee'), get_point(kps1, 'left_ankle'))
    rk_ref = compute_angle(get_point(kps1, 'right_hip'), get_point(kps1, 'right_knee'), get_point(kps1, 'right_ankle'))
    lk_eval = compute_angle(get_point(kps2, 'left_hip'), get_point(kps2, 'left_knee'), get_point(kps2, 'left_ankle'))
    rk_eval = compute_angle(get_point(kps2, 'right_hip'), get_point(kps2, 'right_knee'), get_point(kps2, 'right_ankle'))

    # 차이
    lh_diff = safe_diff(lh_ref, lh_eval)
    rh_diff = safe_diff(rh_ref, rh_eval)
    lk_diff = safe_diff(lk_ref, lk_eval)
    rk_diff = safe_diff(rk_ref, rk_eval)

    # 통과 여부
    pass_lh = pass_check(lh_diff)
    pass_rh = pass_check(rh_diff)
    pass_lk = pass_check(lk_diff)
    pass_rk = pass_check(rk_diff)

    # 유사도
    ref_vec = [lh_ref, rh_ref, lk_ref, rk_ref]
    eval_vec = [lh_eval, rh_eval, lk_eval, rk_eval]
    sim_score = cosine_similarity(ref_vec, eval_vec)

    # 실패 부위
    failed_parts = []
    if not pass_lh: failed_parts.append("left_hip")
    if not pass_rh: failed_parts.append("right_hip")
    if not pass_lk: failed_parts.append("left_knee")
    if not pass_rk: failed_parts.append("right_knee")

    return {
        "left_hip_angle_diff": round(lh_diff, 2) if lh_diff != 999 else None,
        "right_hip_angle_diff": round(rh_diff, 2) if rh_diff != 999 else None,
        "left_knee_angle_diff": round(lk_diff, 2) if lk_diff != 999 else None,
        "right_knee_angle_diff": round(rk_diff, 2) if rk_diff != 999 else None,
        "pass_left_hip": pass_lh,
        "pass_right_hip": pass_rh,
        "pass_left_knee": pass_lk,
        "pass_right_knee": pass_rk,
        "cosine_similarity": round(sim_score, 4),
        "failed_parts": failed_parts
    }

# 영상 처리 함수: 영상에서 프레임 추출
async def extract_frames(video_path: str, output_dir: str, max_frames: int = 87):
    """
    영상에서 프레임을 추출하여 이미지 파일로 저장하는 함수
    Args:
        video_path: 영상 파일 경로
        output_dir: 프레임 이미지 저장 경로
        max_frames: 최대 추출 프레임 수 (기본값: 87)
    Returns:
        추출된 프레임 수
    """
    # 이미지 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"영상 파일을 열 수 없습니다: {video_path}")
    
    # 프레임 수 확인 및 추출 간격 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames) if total_frames > max_frames else 1
    
    # 프레임 추출
    extracted_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0 and extracted_count < max_frames:
            output_image_path = os.path.join(output_dir, f"frame_{extracted_count:03d}.jpg")
            cv2.imwrite(output_image_path, frame)
            extracted_count += 1
        
        frame_idx += 1
    
    cap.release()
    return extracted_count

# 포즈 분석 함수: YOLO 모델을 사용하여 포즈 추정
async def run_pose_estimation(image_dir: str, json_dir: str, model_path: str = YOLO_MODEL_PATH):
    """
    추출된 이미지에서 YOLO 모델을 사용하여 포즈를 추정하고 JSON으로 저장하는 함수
    Args:
        image_dir: 이미지 디렉토리 경로
        json_dir: JSON 저장 디렉토리 경로
        model_path: YOLO 모델 경로
    Returns:
        분석된 이미지 수
    """
    # JSON 저장 디렉토리 생성
    os.makedirs(json_dir, exist_ok=True)
    
    # YOLO 모델 로드
    try:
        yolo_model = YOLO(model_path)
    except Exception as e:
        raise Exception(f"YOLO 모델 로드 실패: {str(e)}")
    
    # 키포인트 이름 정의
    COCO_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                      "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                      "left_wrist", "right_wrist", "left_hip", "right_hip",
                      "left_knee", "right_knee", "left_ankle", "right_ankle"]
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    analyzed_count = 0
    
    # 각 이미지에 대해 포즈 추정 실행
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # YOLO 모델로 포즈 추정
        results = yolo_model(image)
        
        # 결과 JSON 초기화
        json_data = {'image_name': img_file, 'bboxes': [], 'keypoints': []}
        keypoints_dict = {part: {"x": None, "y": None, "confidence": 0.0} for part in COCO_KEYPOINTS}
        
        # 결과 처리
        for result in results:
            # 바운딩 박스 정보 추출
            if result.boxes is not None:
                for bbox, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), 
                                         result.boxes.conf.cpu().numpy(), 
                                         result.boxes.cls.cpu().numpy()):
                    json_data['bboxes'].append({
                        'class': int(cls), 
                        'bbox': list(map(int, bbox)), 
                        'confidence': float(conf)
                    })
            
            # 키포인트 정보 추출
            if result.keypoints is not None:
                for idx, (kp, score) in enumerate(zip(result.keypoints.xy.cpu().numpy()[0], 
                                                    result.keypoints.conf.cpu().numpy()[0])):
                    x, y, conf = int(kp[0]), int(kp[1]), float(score)
                    keypoints_dict[COCO_KEYPOINTS[idx]] = {
                        'x': x if conf > 0.1 else None,
                        'y': y if conf > 0.1 else None,
                        'confidence': conf
                    }
        
        # 키포인트 정보 추가
        json_data["keypoints"] = [
            {"part": part, **keypoints_dict[part]} for part in COCO_KEYPOINTS
        ]
        
        # # 시각화된 이미지 저장 (옵션)
        # if result.keypoints is not None:
        #     visualized_img = results[0].plot()
        #     vis_path = os.path.join(image_dir, f"vis_{img_file}")
        #     cv2.imwrite(vis_path, visualized_img)
        
        # JSON 파일로 저장
        json_path = os.path.join(json_dir, img_file.replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json'))
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(json_data, jf, indent=4)
            
        analyzed_count += 1
    
    return analyzed_count

# 영상 분석 처리 함수: 전체 분석 과정 관리 - LLM 분석 단계 추가
async def process_video(video_path: str, video_id: str, exercise_id: str = None):
    """
    영상을 처리하고 포즈를 분석하는 전체 과정을 관리하는 함수
    Args:
        video_path: 영상 파일 경로
        video_id: 영상 ID
        exercise_id: 운동 ID (squat, pushup, lunge 등), 파일명에서 추출 가능
    Returns:
        분석 결과 정보
    """
    # 운동 ID가 제공되지 않은 경우 파일명에서 추출 시도
    if not exercise_id:
        for ex in exercise_data["exercises"]:
            if ex["id"] in video_id:
                exercise_id = ex["id"]
                break
        
        # 기본값 설정
        if not exercise_id:
            exercise_id = "squat"  # 기본값으로 스쿼트 설정
    
    # 분석 결과 저장 디렉토리 설정
    analysis_dir = os.path.join(VIDEO_ANALYSIS_PATH, video_id)
    image_dir = os.path.join(analysis_dir, "images")
    json_dir = os.path.join(analysis_dir, "json")
    pose_eval_path = os.path.join(analysis_dir, "pose_angle_eval.json")
    llm_result_path = os.path.join(analysis_dir, "analysis_result.json")
    
    # 디렉토리 초기화
    if os.path.exists(analysis_dir):
        shutil.rmtree(analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)
    
    try:
        # 1. 영상에서 프레임 추출
        frame_count = await extract_frames(video_path, image_dir)
        
        # 2. 포즈 추정 실행
        if frame_count > 0:
            analyzed_count = await run_pose_estimation(image_dir, json_dir)
            
            # 3. 포즈 각도 평가
            pose_eval_result = await evaluate_pose_angles(json_dir, exercise_id, pose_eval_path)
            
            # 4. LLM 분석 추가
            llm_analysis_result = await generate_llm_analysis(pose_eval_path, llm_result_path)
            
            # 5. 결과 요약 정보 생성
            summary = {
                "video_id": video_id,
                "exercise_id": exercise_id,
                "frame_count": frame_count,
                "analyzed_count": analyzed_count,
                "pose_evaluation": pose_eval_result,
                "llm_analysis_available": True,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            # 요약 정보 저장
            with open(os.path.join(analysis_dir, "analysis_summary.json"), 'w', encoding='utf-8') as sf:
                json.dump(summary, sf, indent=4)
            
            return summary
        else:
            raise Exception("프레임 추출에 실패했습니다")
        
    except Exception as e:
        # 오류 발생 시 오류 정보 저장
        error_info = {
            "video_id": video_id,
            "exercise_id": exercise_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }
        
        with open(os.path.join(analysis_dir, "analysis_error.json"), 'w', encoding='utf-8') as ef:
            json.dump(error_info, ef, indent=4, ensure_ascii=False)
        
        return error_info

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["start_time"] = datetime.now()
    app_state["last_connection_cleanup"] = datetime.now()
    
    # 분석 결과 디렉토리 생성
    os.makedirs(VIDEO_ANALYSIS_PATH, exist_ok=True)
    os.makedirs(REFERENCE_POSES_PATH, exist_ok=True)
    
    yield
    app_state["is_running"] = False

app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 디렉토리 생성 (없는 경우)
if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY, exist_ok=True)

# 비디오 저장 디렉토리 생성 (없는 경우)
if not os.path.exists(VIDEO_STORAGE_PATH):
    os.makedirs(VIDEO_STORAGE_PATH, exist_ok=True)
    # os.makedirs(os.path.join(VIDEO_STORAGE_PATH, "uploads"), exist_ok=True)

# 정적 파일 서빙 설정
app.mount("/data", StaticFiles(directory=DATA_DIRECTORY), name="data")
app.mount("/video_uploads", StaticFiles(directory=VIDEO_STORAGE_PATH), name="video_uploads")

# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    uptime = datetime.now() - app_state["start_time"]
    return {
        "status": "healthy" if app_state["is_running"] else "shutting_down",
        "connected_cameras": len(camera_info),
        "active_websockets": len(active_connections),
        "uptime_seconds": uptime.total_seconds(),
        "uptime_formatted": str(uptime)
    }

# 서버 시간 엔드포인트
@app.get("/server_time")
async def get_server_time():
    """서버의 현재 시간 정보 제공"""
    now = datetime.now()
    return {
        "server_time": now.isoformat(),
        "timestamp": time.time()
    }

# 카메라 목록 조회 엔드포인트
@app.get("/cameras")
async def get_cameras():
    """등록된 카메라 목록 조회"""
    with data_lock:
        # 현재 연결된 카메라만 반환 (WebSocket이 있는 카메라)
        active_cameras = [
            camera_id for camera_id, info in camera_info.items()
            if "websocket" in info
        ]
    
    return {"cameras": active_cameras, "count": len(active_cameras)}

# 운동 관련 엔드포인트
@app.get("/exercises")
async def get_exercises():
    """사용 가능한 운동 목록 조회"""
    return {"exercises": exercise_data["exercises"]}

@app.get("/exercise/{exercise_id}")
async def get_exercise_detail(exercise_id: str):
    """특정 운동의 상세 정보 조회"""
    # 운동 ID로 운동 찾기
    exercise = None
    for exercise_item in exercise_data["exercises"]:
        if exercise_item["id"] == exercise_id:
            exercise = exercise_item
            break
    
    if not exercise:
        raise HTTPException(status_code=404, detail="운동을 찾을 수 없습니다")
    
    return exercise

# 카메라 연결 해제 처리 함수
async def disconnect_camera(camera_id: str):
    """카메라 연결 해제 및 리소스 정리"""
    with data_lock:
        if camera_id in camera_info:
            # 연결된 WebSocket 종료
            if "websocket" in camera_info[camera_id]:
                try:
                    await camera_info[camera_id]["websocket"].close(code=1000)
                except Exception:
                    pass
            
            # 구독자에게 카메라 연결 해제 알림
            subscribers = camera_info[camera_id].get("subscribers", set()).copy()
            for ws in subscribers:
                try:
                    await ws.send_json({
                        "type": "camera_disconnected",
                        "camera_id": camera_id,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception:
                    pass
            
            # 카메라 정보 완전히 삭제
            del camera_info[camera_id]
            
            # 이미지 데이터도 삭제
            if camera_id in latest_image_data:
                del latest_image_data[camera_id]
            if camera_id in latest_timestamps:
                del latest_timestamps[camera_id]
            
            print(f"카메라 {camera_id} 연결 해제 및 정리 완료")
            return True
    return False

# 정기적인 연결 정리 작업
async def cleanup_connections():
    # 60초마다 실행
    while app_state["is_running"]:
        try:
            now = datetime.now()
            
            # 마지막 정리 후 60초 이상 지났는지 확인
            if (now - app_state["last_connection_cleanup"]).total_seconds() >= 60:
                # 일반 WebSocket 연결 확인
                dead_connections = set()
                for ws in active_connections:
                    try:
                        await ws.send_text("ping")
                    except Exception:
                        dead_connections.add(ws)
                
                # 일반 WebSocket 데드 연결 제거
                for ws in dead_connections:
                    active_connections.discard(ws)
                
                # 카메라 연결 확인 및 정리
                disconnected_cameras = []
                with data_lock:
                    for camera_id, info in list(camera_info.items()):
                        # 마지막 활동 시간이 60초 이상 지난 카메라 확인
                        if "last_seen" in info and (now - info["last_seen"]).total_seconds() >= 60:
                            # 연결 종료 후 정보 삭제
                            disconnected_cameras.append(camera_id)
                
                # 비동기 컨텍스트 밖에서 실행
                for camera_id in disconnected_cameras:
                    await disconnect_camera(camera_id)
                
                # 정리 완료 시간 업데이트
                app_state["last_connection_cleanup"] = now
            
            await asyncio.sleep(10)  # 10초마다 확인
        except Exception:
            await asyncio.sleep(10)  # 오류 발생해도 계속 실행

# WebSocket 구독자에게 이미지 브로드캐스트
async def broadcast_image_to_subscribers(camera_id: str, image_data: str, timestamp: datetime):
    """WebSocket 구독자들에게 이미지 데이터 직접 전송"""
    if camera_id not in camera_info or "subscribers" not in camera_info[camera_id]:
        return
    
    # 메시지 준비
    message = {
        "type": "image",
        "camera_id": camera_id,
        "image_data": image_data,
        "timestamp": timestamp.isoformat()
    }
    
    # 직렬화
    message_str = json.dumps(message)
    
    # 구독자 목록 복사 (비동기 처리 중 변경될 수 있음)
    with data_lock:
        if camera_id in camera_info and "subscribers" in camera_info[camera_id]:
            subscribers = camera_info[camera_id]["subscribers"].copy()
        else:
            return
    
    # 끊어진 연결 추적
    dead_connections = set()
    
    # 모든 구독자에게 전송
    for websocket in subscribers:
        try:
            await websocket.send_text(message_str)
        except Exception:
            dead_connections.add(websocket)
    
    # 끊어진 연결 정리
    if dead_connections:
        with data_lock:
            if camera_id in camera_info and "subscribers" in camera_info[camera_id]:
                camera_info[camera_id]["subscribers"] -= dead_connections

# 웹소켓 클라이언트 알림 함수
async def notify_clients(camera_id: str):
    """웹소켓 클라이언트에게 이미지 업데이트 알림"""
    if not active_connections:
        return
        
    # 메시지 준비
    message = {
        "type": "image_update",
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat()
    }
    
    message_str = json.dumps(message)
    
    # 연결된 모든 클라이언트에게 알림
    dead_connections = set()
    for websocket in active_connections:
        try:
            await websocket.send_text(message_str)
        except Exception:
            dead_connections.add(websocket)
    
    # 끊어진 연결 정리
    if dead_connections:
        for dead in dead_connections:
            active_connections.discard(dead)
        
        # 상태 업데이트
        app_state["active_websockets"] = len(active_connections)

# 웹소켓 연결 유지 함수
async def keep_websocket_alive(websocket: WebSocket):
    """WebSocket 연결을 유지하는 함수"""
    ping_interval = 30  # 30초마다 핑 전송
    last_ping_time = time.time()
    last_received_time = time.time()
    max_idle_time = 60  # 60초 동안 응답이 없으면 연결 종료
    
    try:
        while True:
            current_time = time.time()
            
            # 마지막 응답으로부터 너무 오래 경과했는지 확인
            if current_time - last_received_time > max_idle_time:
                # 연결이 너무 오래 idle 상태임
                return False
            
            # 정기적인 핑 전송
            if current_time - last_ping_time >= ping_interval:
                try:
                    # 핑 메시지 전송
                    await websocket.send_text("ping")
                    last_ping_time = current_time
                except Exception:
                    # 핑 전송 실패
                    return False
            
            # 메시지 수신 시도 (짧은 타임아웃으로 반응성 유지)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                last_received_time = time.time()  # 메시지 수신 시간 업데이트
                
                # 핑/퐁 처리
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "pong":
                    # 클라이언트에서 보낸 퐁 응답
                    pass
            except asyncio.TimeoutError:
                # 타임아웃은 정상적인 상황, 계속 진행
                pass
            except Exception:
                # 기타 오류 발생 시 연결 종료
                return False
            
            # 잠시 대기 후 다음 루프
            await asyncio.sleep(0.1)
    except Exception:
        return False
    
    return True

# 웹소켓 연결 엔드포인트
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결 처리"""
    await websocket.accept()
    
    # 연결 목록에 추가
    active_connections.add(websocket)
    
    # 상태 업데이트
    app_state["active_websockets"] = len(active_connections)
    
    # 초기 카메라 목록 전송
    with data_lock:
        active_cameras = list(camera_info.keys())
    
    try:
        # 초기 데이터 전송
        await websocket.send_json({
            "type": "init",
            "cameras": active_cameras,
            "timestamp": datetime.now().isoformat()
        })
        
        # 연결 유지 루프 - 개선된 함수 사용
        if not await keep_websocket_alive(websocket):
            # 연결 유지 실패
            pass
    except Exception:
        # 오류 처리 - 조용히 진행
        pass
    finally:
        # 연결 목록에서 제거
        active_connections.discard(websocket)
        # 상태 업데이트
        app_state["active_websockets"] = len(active_connections)

# 웹캠 카메라용 WebSocket 엔드포인트
@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """웹캠 클라이언트의 WebSocket 연결 처리"""
    await websocket.accept()
    
    camera_id = None
    try:
        # 첫 메시지에서 카메라 ID 확인 또는 생성
        first_message = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        data = json.loads(first_message)
        
        if data.get("type") == "register":
            camera_id = data.get("camera_id")
            
            # 기존 동일 ID 카메라가 있으면 연결 해제
            if camera_id in camera_info:
                await disconnect_camera(camera_id)
        
        # 새 카메라 ID 생성
        if not camera_id:
            camera_id = f"webcam_{len(camera_info) + 1}"
        
        # 카메라 정보 저장
        with data_lock:
            camera_info[camera_id] = {
                "info": data.get("info", {}),
                "last_seen": datetime.now(),
                "websocket": websocket,
                "subscribers": set(),  # 구독자 목록 초기화
                "status": data.get("status", "on")  # 클라이언트가 제공한 상태 또는 기본값 'on'
            }
        
        # 카메라에 ID 전송
        await websocket.send_json({
            "type": "connection_successful",
            "camera_id": camera_id
        })
        
        print(f"웹캠 연결됨: {camera_id}, 상태: {camera_info[camera_id].get('status', 'on')}")
        
        # 연결 유지 및 프레임 수신 루프
        last_seen = datetime.now()
        last_keepalive = time.time()
        keepalive_interval = 15  # 15초마다 핑 전송
        
        while True:
            try:
                # 짧은 타임아웃으로 메시지 수신 대기
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                
                # 핑/퐁 처리
                if message == "ping":
                    await websocket.send_text("pong")
                    last_seen = datetime.now()
                    continue
                elif message == "pong":
                    last_seen = datetime.now()
                    continue
                
                # JSON 메시지 파싱
                try:
                    data = json.loads(message)
                    
                    # 타입별 메시지 처리
                    msg_type = data.get("type")
                    
                    if msg_type == "frame":
                        # 프레임 저장
                        image_data = data.get("image_data")
                        if image_data:
                            timestamp = datetime.now()
                            last_seen = timestamp
                            
                            # 이미지 저장
                            with data_lock:
                                latest_image_data[camera_id] = image_data
                                latest_timestamps[camera_id] = timestamp
                                
                                # 카메라 상태 업데이트
                                if camera_id in camera_info:
                                    camera_info[camera_id]["last_seen"] = timestamp
                            
                            # 구독자에게 이미지 직접 전송
                            await broadcast_image_to_subscribers(camera_id, image_data, timestamp)
                            
                            # 일반 웹소켓 클라이언트에게 알림
                            await notify_clients(camera_id)
                    
                    elif msg_type == "disconnect":
                        # 클라이언트에서 종료 요청 - 정상 종료
                        print(f"카메라 {camera_id}에서 연결 종료 요청을 받음")
                        break
                    
                    elif msg_type == "status_changed":
                        # 카메라 상태 변경 메시지
                        new_status = data.get("status")
                        if new_status and camera_id in camera_info:
                            with data_lock:
                                camera_info[camera_id]["status"] = new_status
                                print(f"카메라 {camera_id} 상태 업데이트: {new_status}")
                    
                    elif msg_type == "recording_completed":
                        # 녹화 완료 메시지 처리
                        video_id = data.get("video_id")
                        video_path = data.get("video_path")
                        
                        # 녹화 정보 저장
                        with data_lock:
                            if camera_id in camera_info:
                                if "recordings" not in camera_info[camera_id]:
                                    camera_info[camera_id]["recordings"] = []
                                
                                # 녹화 기록 추가
                                camera_info[camera_id]["recordings"].append({
                                    "video_id": video_id,
                                    "path": video_path,
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                print(f"카메라 {camera_id}의 녹화 정보 저장 완료: {video_id} - {datetime.now().isoformat()}")
                        
                except json.JSONDecodeError:
                    # JSON 파싱 오류는 무시
                    pass
                
                # 정기적인 핑 전송
                current_time = time.time()
                if current_time - last_keepalive >= keepalive_interval:
                    try:
                        await websocket.send_text("ping")
                        last_keepalive = current_time
                    except Exception:
                        # 핑 전송 실패 시 연결 종료
                        break
                    
            except asyncio.TimeoutError:
                # 타임아웃은 정상적인 상황, 핑 체크만 수행
                current_time = time.time()
                if current_time - last_keepalive >= keepalive_interval:
                    try:
                        await websocket.send_text("ping")
                        last_keepalive = current_time
                    except Exception:
                        # 핑 전송 실패 시 연결 종료
                        break
                
                # 장시간 메시지가 없는지 확인 (45초 이상)
                if (datetime.now() - last_seen).total_seconds() > 45:
                    # 너무 오래 메시지가 없으면 연결 종료
                    print(f"카메라 {camera_id} 45초 동안 활동 없음, 연결 종료")
                    break
            except Exception as e:
                # 기타 예외 발생 시 연결 종료
                print(f"카메라 {camera_id} 처리 오류: {str(e)}")
                break
    except Exception as e:
        print(f"웹캠 웹소켓 오류: {str(e)}")
    finally:
        # 연결 종료 처리 - 완전히 삭제
        if camera_id:
            print(f"카메라 {camera_id} 연결 종료 처리 중...")
            await disconnect_camera(camera_id)

# WebSocket을 통한 이미지 스트리밍 엔드포인트
@app.websocket("/ws/stream/{camera_id}")
async def stream_camera(websocket: WebSocket, camera_id: str):
    """특정 카메라의 이미지를 WebSocket으로 스트리밍"""
    await websocket.accept()
    
    # 해당 카메라가 존재하는지 확인
    if camera_id not in camera_info:
        await websocket.close(code=1008, reason=f"카메라 ID {camera_id}를 찾을 수 없습니다")
        return
    
    # 해당 카메라의 실시간 스트리밍을 구독하는 클라이언트 등록
    with data_lock:
        if "subscribers" not in camera_info[camera_id]:
            camera_info[camera_id]["subscribers"] = set()
        
        camera_info[camera_id]["subscribers"].add(websocket)
    
    try:
        # 최신 이미지가 있으면 즉시 전송
        with data_lock:
            if camera_id in latest_image_data and camera_id in latest_timestamps:
                image_data = latest_image_data[camera_id]
                timestamp = latest_timestamps[camera_id]
                
                if image_data:
                    # 이미지 메시지 전송
                    await websocket.send_json({
                        "type": "image",
                        "camera_id": camera_id,
                        "image_data": image_data,
                        "timestamp": timestamp.isoformat()
                    })
        
        # 연결 유지 루프 - 개선된 함수 사용
        if not await keep_websocket_alive(websocket):
            # 연결 유지 실패
            pass
    except Exception:
        # 예외 처리 - 조용히 진행
        pass
    finally:
        # 구독 목록에서 제거
        with data_lock:
            if camera_id in camera_info and "subscribers" in camera_info[camera_id]:
                camera_info[camera_id]["subscribers"].discard(websocket)

# 정리 작업 백그라운드 태스크 시작
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_connections())

# 카메라 상태 제어 엔드포인트
@app.post("/cameras/{camera_id}/status")
async def camera_status_control(
    camera_id: str,
    status: str = Body(..., embed=True)
):
    """카메라 상태 제어 (off, on, mask, ready, record, detect)"""
    valid_statuses = ["off", "on", "mask", "ready", "record", "detect"]
    
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400, 
            detail=f"유효하지 않은 상태입니다. 유효한 상태: {', '.join(valid_statuses)}"
        )
    
    with data_lock:
        if camera_id not in camera_info:
            raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다")
        
        if "websocket" not in camera_info[camera_id]:
            raise HTTPException(status_code=400, detail="카메라가 현재 연결되어 있지 않습니다")
        
        try:
            # 카메라 클라이언트에 명령 전송
            websocket = camera_info[camera_id]["websocket"]
            await websocket.send_json({
                "type": "status_control",
                "status": status
            })
            
            # 카메라 상태 업데이트
            camera_info[camera_id]["status"] = status
            
            # 녹화 시작 시간 기록
            if status == "record":
                camera_info[camera_id]["recording_start_time"] = datetime.now().isoformat()
            
            # ready 상태로 변경된 경우 모든 카메라 상태 확인
            if status == "ready":
                all_ready = all(info.get("status") == "ready" for _, info in camera_info.items() if "websocket" in info)
                print("all_ready", all_ready)
                # 모든 카메라가 ready 상태면 record로 변경
                if all_ready:
                    for cam_id, info in camera_info.items():
                        if "websocket" in info:
                            await info["websocket"].send_json({"type": "status_control", "status": "record"})
                            info["status"] = "record"
                            info["recording_start_time"] = datetime.now().isoformat()
            
            return {
                "status": "success",
                "camera_id": camera_id,
                "camera_status": status,
                "message": f"카메라 상태가 '{status}'로 변경되었습니다"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"카메라 제어 중 오류 발생: {str(e)}")

# 카메라 상태 조회 엔드포인트
@app.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str):
    """카메라 상태 조회"""
    with data_lock:
        if camera_id not in camera_info:
            raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다")
        
        # 기본 응답 정보
        response = {
            "camera_id": camera_id,
            "connected": "websocket" in camera_info[camera_id],
            "status": camera_info[camera_id].get("status", "off")
        }
        
        # 녹화 중인 경우 추가 정보
        if response["status"] == "record" and "recording_start_time" in camera_info[camera_id]:
            start_time = camera_info[camera_id]["recording_start_time"]
            response["recording"] = {
                "start_time": start_time,
                "duration_seconds": (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
            }
        
        return response

# --- 미디어 관리 및 분석 관련 엔드포인트 ---

# # 사용자 업로드 영상 스트리밍 엔드포인트
# @app.get("/uploaded_videos/{video_id}")
# async def get_uploaded_video(video_id: str):
#     """사용자가 업로드한 영상을 스트리밍하여 제공"""
#     # video_path = os.path.join(VIDEO_STORAGE_PATH, "uploads", f"{video_id}.mp4")
#     video_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_id}.mp4")
    
#     if not os.path.exists(video_path):
#         raise HTTPException(status_code=404, detail="업로드된 영상을 찾을 수 없습니다")
    
#     return FileResponse(
#         video_path,
#         media_type="video/mp4",
#         filename=f"user_video_{video_id}.mp4"
#     )
@app.get("/uploaded_videos_name")
async def get_uploaded_videos_name():
    """사용자가 업로드한 영상 목록 조회"""
    video_dir = VIDEO_STORAGE_PATH
    if not os.path.exists(video_dir):
        return []
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    return video_files

# 영상 업로드 엔드포인트
@app.post("/videos/upload")
async def upload_exercise_video(
    video: UploadFile = File(...),
    exercise_id: str = Form(...),
    user_id: Optional[str] = Form(None),
    camera_id: Optional[str] = Form(None),
    analyze: bool = Form(True)  # 업로드 후 바로 분석할지 여부
):
    """운동 영상 업로드 및 분석 요청"""
    # 업로드 디렉토리 생성
    upload_dir = VIDEO_STORAGE_PATH
    os.makedirs(upload_dir, exist_ok=True)
    
    # 파일 ID 생성 (카메라 ID가 있으면 사용)
    if camera_id:
        timestamp = int(time.time())
        video_id = f"{camera_id}_{timestamp}"
    else:
        video_id = f"{int(time.time())}_{exercise_id}"
    
    file_path = os.path.join(upload_dir, f"{video_id}.mp4")
    
    # 파일 저장
    try:
        with open(file_path, "wb") as buffer:
            contents = await video.read()
            buffer.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류 발생: {str(e)}")
    
    # 카메라 정보가 있는 경우 녹화 정보 추가
    if camera_id and camera_id in camera_info:
        with data_lock:
            if "recordings" not in camera_info[camera_id]:
                camera_info[camera_id]["recordings"] = []
            
            # 녹화 기록 추가
            camera_info[camera_id]["recordings"].append({
                "video_id": video_id,
                "path": file_path,
                "timestamp": datetime.now().isoformat(),
                "exercise_id": exercise_id
            })
    
    # 분석 요청이 있는 경우 비동기로 분석 작업 시작
    analysis_task = None
    if analyze:
        # 비동기 태스크로 분석 작업 실행
        analysis_task = asyncio.create_task(process_video(file_path, video_id, exercise_id))
    
    response_data = {
        "video_id": video_id,
        "camera_id": camera_id,
        "exercise_id": exercise_id,
        "status": "uploaded",
        "message": "영상이 업로드되었습니다.",
        "video_url": f"/uploaded_videos/{video_id}"
    }
    
    # 분석 요청이 있는 경우 메시지 추가
    if analyze:
        response_data["analysis_status"] = "processing"
        response_data["message"] += " 분석이 진행 중입니다."
        response_data["analysis_url"] = f"/analysis/video/{video_id}"
    
    return response_data

# 영상 분석 요청 엔드포인트
@app.post("/videos/{video_id}/analyze")
async def request_video_analysis(video_id: str, exercise_id: Optional[str] = None):
    """기존 업로드된 영상 분석 요청"""
    # 영상 파일 경로 확인
    video_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_id}.mp4")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="업로드된 영상을 찾을 수 없습니다")
    
    # 비동기 태스크로 분석 작업 실행
    asyncio.create_task(process_video(video_path, video_id, exercise_id))
    
    return {
        "video_id": video_id,
        "exercise_id": exercise_id,
        "status": "processing",
        "message": "영상 분석이 시작되었습니다.",
        "analysis_url": f"/analysis/video/{video_id}"
    }

# 영상 분석 결과 조회 엔드포인트
@app.get("/analysis/video/{video_id}")
async def get_video_analysis_result(video_id: str):
    """영상 분석 결과 조회"""
    # 분석 결과 디렉토리 확인
    analysis_dir = os.path.join(VIDEO_ANALYSIS_PATH, video_id)
    
    if not os.path.exists(analysis_dir):
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다")
    
    # 요약 파일 확인
    summary_path = os.path.join(analysis_dir, "analysis_summary.json")
    error_path = os.path.join(analysis_dir, "analysis_error.json")
    result_path = os.path.join(analysis_dir, "analysis_result.json")
    pose_eval_path = os.path.join(analysis_dir, "pose_angle_eval.json")
    
    # 결과 객체 초기화
    response = {
        "video_id": video_id,
        "status": "unknown"
    }
    
    # 에러 정보가 있는 경우
    if os.path.exists(error_path):
        with open(error_path, 'r', encoding='utf-8') as f:
            error_info = json.load(f)
        response.update(error_info)
        response["status"] = "failed"
        return response
    
    # 요약 정보가 있는 경우
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        response.update(summary)
    
    # 포즈 평가 결과가 있는 경우
    if os.path.exists(pose_eval_path):
        with open(pose_eval_path, 'r', encoding='utf-8') as f:
            pose_data = json.load(f)
        response["pose_evaluation_details"] = pose_data.get("summary", {})
    
    # LLM 분석 결과가 있는 경우
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            llm_result = json.load(f)
        response["llm_analysis"] = llm_result
        response["status"] = "completed"
    else:
        # LLM 분석 결과가 없는 경우
        response["llm_analysis_available"] = False
        response["status"] = "processing"
        response["message"] = "LLM 분석이 진행 중이거나 아직 결과가 없습니다."
    
    return response
        

# # 녹화 목록 조회 엔드포인트
# @app.get("/cameras/{camera_id}/recordings")
# async def get_camera_recordings(camera_id: str):
#     """카메라의 녹화 목록 조회"""
#     with data_lock:
#         if camera_id not in camera_info:
#             raise HTTPException(status_code=404, detail="카메라를 찾을 수 없습니다")
        
#         # 녹화 목록 반환
#         recordings = camera_info[camera_id].get("recordings", [])
        
#         return {
#             "camera_id": camera_id,
#             "recordings_count": len(recordings),
#             "recordings": recordings
#         }

# # 녹화 비디오 스트리밍 엔드포인트
# @app.get("/cameras/{camera_id}/recordings/{video_id}")
# async def stream_recording(camera_id: str, video_id: str):
#     """카메라의 녹화 비디오 스트리밍"""
#     # 업로드된 비디오 경로 생성
#     # video_path = os.path.join(VIDEO_STORAGE_PATH, "uploads", f"{video_id}.mp4")
#     video_path = os.path.join(VIDEO_STORAGE_PATH, f"{video_id}.mp4")
    
#     if not os.path.exists(video_path):
#         raise HTTPException(status_code=404, detail="녹화 비디오를 찾을 수 없습니다")
    
#     return FileResponse(
#         video_path,
#         media_type="video/mp4",
#         filename=f"recording_{camera_id}_{video_id}.mp4"
#     )

# 서버 실행 (직접 실행 시)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 