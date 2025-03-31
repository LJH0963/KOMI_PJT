# 라이브러리 불러오기
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re

# 환경설정
from dotenv import load_dotenv
load_dotenv()

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