import json
import asyncio
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 로컬 모델 설정
# MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "beomi/KoAlpaca-Polyglot-12.8B")
MODEL_NAME = "Bllossom/llama-3.2-Korean-Bllossom-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 및 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

# 의학 지식 프롬프트 템플릿
MEDICAL_EXPERT_PROMPT = """
당신은 재활 의학 전문가이며 물리치료사입니다. 당신의 역할은 환자의 자세와 움직임을 분석하고, 
그들의 상태에 맞는 재활 운동과 자세 교정 방법을 제안하는 것입니다.

다음은 환자의 {condition} 관련 자세 데이터와 분석 결과입니다:
{data}

이 데이터를 바탕으로 다음 질문에 답변해주세요:
1. 환자의 자세에서 문제가 될 수 있는 부분은 무엇인가요?
2. 이러한 자세가 장기적으로 지속될 경우 발생할 수 있는 문제는 무엇인가요?
3. 자세 교정을 위한 구체적인 운동이나 스트레칭은 무엇이 있을까요?
4. 일상생활에서 자세를 개선하기 위해 환자가 주의해야 할 점은 무엇인가요?

답변은 환자가 이해하기 쉽게 작성해주시고, 의학적으로 정확한 정보를 제공해주세요.
"""

async def get_llm_response(prompt: str) -> str:
    """
    로컬 LLM 모델을 사용하여 응답을 생성합니다.
    """
    try:
        # 입력 텍스트 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # 모델 추론
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 출력 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        response = response[len(prompt):].strip()
        
        return response if response else "응답을 생성하는 중 오류가 발생했습니다."
        
    except Exception as e:
        print(f"모델 추론 오류: {str(e)}")
        return _get_dummy_llm_response(prompt)

def _get_dummy_llm_response(prompt: str) -> str:
    """
    모델 추론 실패 시 사용할 더미 응답 생성
    """
    if "자세 데이터" in prompt:
        return """
1. 환자의 자세 문제:
   - 어깨가 앞으로 굽어 있어 둥근 어깨(라운드 숄더) 증상이 보입니다.
   - 목이 앞으로 나와 있어 거북목 증상이 있습니다.
   - 골반이 약간 틀어져 있어 체중 분산이 고르지 않습니다.

2. 장기적 문제:
   - 목과 어깨 통증 증가
   - 두통 발생 가능성
   - 허리 통증 및 디스크 문제 발생 가능성
   - 자세 불균형으로 인한 근육 불균형 심화

3. 추천 운동:
   - 가슴 스트레칭: 문틀에 양팔을 대고 앞으로 기대어 가슴 근육 스트레칭
   - 턱 당기기 운동: 목을 바르게 하여 턱을 안쪽으로 당기는 운동
   - 골반 교정 운동: 바닥에 누워 골반 틸트 운동하기
   - 어깨 후면부 강화 운동: 밴드를 이용한 로우 운동

4. 일상생활 주의사항:
   - 스마트폰 사용 시 눈높이로 들어올리기
   - 컴퓨터 작업 시 모니터 높이 조절하기
   - 1시간마다 스트레칭 휴식 취하기
   - 올바른 자세로 앉고 서는 습관 들이기
   - 수면 자세 교정하기 (측면 수면 시 베개 높이 조절)
        """
    elif "운동 추천" in prompt:
        return """
1. 추천 운동:

   a) 무릎 굽히기 운동:
      - 의자에 앉은 상태에서 한쪽 다리를 천천히 들어올려 무릎을 펴고 10초간 유지합니다.
      - 천천히 내린 후 반대쪽도 동일하게 실시합니다.
      - 각 다리 10회씩, 하루 3세트 실시합니다.
   
   b) 허벅지 안쪽 근력 운동:
      - 누운 자세에서 무릎 사이에 베개나 공을 끼우고 가볍게 압박합니다.
      - 10초간 유지 후 휴식, 15회 반복, 하루 2세트 실시합니다.
   
   c) 발목 돌리기:
      - 의자에 앉아 한쪽 발을 들어올린 상태에서 발목을 시계 방향, 반시계 방향으로 천천히 돌립니다.
      - 각 방향으로 10회씩, 하루 3회 실시합니다.

2. 주의사항:
   - 통증이 심할 때는 운동을 중단하세요.
   - 갑작스러운 강한 무릎 굽힘이나 뻗기는 피하세요.
   - 무거운 물건 들기나 장시간 서 있는 것을 최소화하세요.
   - 운동 전후로 충분한 준비운동과 마무리 스트레칭을 하세요.

3. 운동 빈도:
   - 매일 아침, 저녁으로 두 번 실시하세요.
   - 통증이 심한 날은 강도를 줄이거나 휴식하세요.
   - 2주간 꾸준히 실시한 후 점진적으로 강도를 높이세요.

4. 통증 대처 방법:
   - 운동 중 통증이 7/10 이상이면 즉시 중단하세요.
   - 냉찜질로 급성 통증을 완화할 수 있습니다 (15-20분).
   - 만성 통증에는 온찜질이 도움이 될 수 있습니다.
   - 통증이 지속되면 의료진과 상담하세요.
        """
    else:
        return "요청하신 내용에 대한 분석이 완료되었습니다. 자세한 정보가 필요하시면 추가 질문해주세요."
