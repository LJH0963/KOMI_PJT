# 라이브러리 불러오기
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from LLM import load_llm, format_docs, format_response

# 환경설정
from dotenv import load_dotenv
load_dotenv()

# 검색기 설정
retriever = 

# LLM 모델설정
llm = load_llm()

# Prompt 설정
prompt = 

# Langchain 설정
rag_chain = (
    RunnableMap({"context": retriever | format_docs, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(format_response)
)