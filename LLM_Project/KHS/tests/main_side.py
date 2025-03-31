# 라이브러리 불러오기
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 각 파일의 함수 불러오기
from LLM import load_llm, format_docs, format_response
from vectorstore import Vectorstore
from prompt import generate_summary_prompt

# 환경설정
from dotenv import load_dotenv
load_dotenv()

def main():
    # LLM 모델설정('gpt-4o-mini' 자동설정)
    llm = load_llm()

    # 검색기 설정
    vectorstore = Vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",           # 유사도 분석 기반
        search_kwargs={"k": 5}              # 3개의 문장만 갖고오기
        )

    # 자연어 Prompt 생성
    input_path = "../../LJH/output_json/side_pose_angle_eval.json"
    question = generate_summary_prompt(input_path)

    # Template 설정
    prompt_template = PromptTemplate.from_template(
        "{context}\n\n{question}")

    # Langchain 설정
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

    response = rag_chain.invoke(question)
    print("\n운동 분석 결과 요약: ")
    print(response)

if __name__ == "__main__":
    main()