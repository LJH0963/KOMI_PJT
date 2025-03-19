import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

# 의학 지식 프롬프트 템플릿
MEDICAL_EXPERT_PROMPT = """
당신은 재활 의학 전문가이며 물리치료사입니다. 다음 자세 데이터를 바탕으로 분석해주세요.
"""

EXERCISE_RECOMMENDATION_PROMPT = """
당신은 재활 의학 전문가이며 물리치료사입니다. 다음 환자 정보를 바탕으로 적절한 운동 추천을 제공해주세요.
"""

# 더미 응답 템플릿
POSTURE_ANALYSIS_RESPONSE = """
1. 환자의 자세 문제:
   - 어깨가 앞으로 굽어 있어 둥근 어깨(라운드 숄더) 증상이 보입니다.
   - 목이 앞으로 나와 있어 거북목 증상이 있습니다.

2. 장기적 문제:
   - 목과 어깨 통증 증가
   - 두통 발생 가능성

3. 추천 운동:
   - 가슴 스트레칭: 문틀에 양팔을 대고 앞으로 기대어 가슴 근육 스트레칭
   - 턱 당기기 운동: 목을 바르게 하여 턱을 안쪽으로 당기는 운동
"""

EXERCISE_RECOMMENDATION_RESPONSE = """
1. 추천 운동:
   a) 무릎 굽히기 운동
   b) 허벅지 안쪽 근력 운동
   c) 발목 돌리기

2. 주의사항:
   - 통증이 심할 때는 운동을 중단하세요.
   - 갑작스러운 강한 무릎 굽힘이나 뻗기는 피하세요.
"""

DEFAULT_RESPONSE = "요청하신 내용에 대한 더미 분석 결과입니다. 실제 LLM 연동 시 자세한 분석이 제공됩니다."

async def get_llm_response(prompt: str) -> str:
    """
    더미 LLM 응답을 생성하는 함수
    """
    print(f"LLM 요청: {prompt[:50]}...")
    
    # 실제 LLM 호출 대신 더미 응답 반환
    await asyncio.sleep(1)  # 실제 API 호출 지연 시뮬레이션
    return _get_dummy_llm_response(prompt)

def _get_dummy_llm_response(prompt: str) -> str:
    """
    더미 LLM 응답 생성
    """
    if "자세 데이터" in prompt:
        return POSTURE_ANALYSIS_RESPONSE
    elif "운동 추천" in prompt:
        return EXERCISE_RECOMMENDATION_RESPONSE
    else:
        return DEFAULT_RESPONSE

async def get_llm_analysis(session_data: List[Dict]) -> Dict:
    """
    세션 데이터를 분석하여 LLM 기반 분석 결과 제공 (더미 함수)
    """
    # 세션 데이터 요약
    data_summary = {
        "session_duration": f"{len(session_data)} 프레임 분석",
        "average_accuracy": "75.5%",
        "exercise_types": ["shoulder", "posture"],
        "sample_frames": len(session_data)
    }
    
    # 더미 LLM 응답 생성
    prompt = f"자세 데이터: {json.dumps(data_summary)}"
    analysis_text = await get_llm_response(prompt)
    
    # 분석 결과 구조화
    analysis_result = {
        "session_summary": data_summary,
        "posture_analysis": analysis_text.strip(),
        "timestamp": datetime.now().isoformat()
    }
    
    return analysis_result

async def get_exercise_recommendation(
    medical_condition: str,
    pain_level: int,
    previous_exercise: Optional[str] = None
) -> Dict:
    """
    운동 추천 더미 함수
    """
    # 더미 응답 생성
    prompt = f"운동 추천: {medical_condition}, 통증 수준: {pain_level}"
    recommendation_text = await get_llm_response(prompt)
    
    # 분석 결과 구조화
    recommendation = {
        "recommendation_text": recommendation_text.strip(),
        "medical_condition": medical_condition,
        "pain_level": pain_level,
        "timestamp": datetime.now().isoformat()
    }
    
    return recommendation
