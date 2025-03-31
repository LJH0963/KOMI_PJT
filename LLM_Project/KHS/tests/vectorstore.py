# 라이브러리 불러오기
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 환경설정
from dotenv import load_dotenv
load_dotenv()

# OpenAI Embeddings()
embedding_model = OpenAIEmbeddings()

# 저장된 Chroma 벡터 DB 로드
def Vectorstore():
    return Chroma(
    persist_directory="../chroma_db/pdf_docs",
    embedding_function=embedding_model
    )