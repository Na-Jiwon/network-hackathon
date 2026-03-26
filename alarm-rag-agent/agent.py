import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from collections import Counter

load_dotenv()


# 기존 인덱스 로드, 없으면 생성
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    if os.path.exists("faiss_index"):
        return FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    df = pd.read_csv("data/Q2_train.csv")
    df_unique = df[['alarmmsg_original', 'root_cause_type']].drop_duplicates()
    docs = []
    labels = []
    for _, row in df_unique.iterrows():
        docs.append(row['alarmmsg_original'])
        labels.append(row['root_cause_type'])
    vectorstore = FAISS.from_texts(
        docs, embeddings,
        metadatas=[{"label": l} for l in labels]
    )
    vectorstore.save_local("faiss_index")
    return vectorstore


def classify_alarm(vectorstore, alarm_message, agent):
    """하이브리드 분류: FAISS 만장일치 시 바로 분류, 불확실 시 Agent 판단"""
    results = vectorstore.similarity_search_with_score(alarm_message, k=3)
    filtered = [(doc, score) for doc, score in results if score < 0.8]

    if filtered:
        labels = [doc.metadata.get("label") for doc, _ in filtered]
        vote = Counter(labels)

        # 만장일치: FAISS 결과를 바로 신뢰
        if len(vote) == 1:
            label = vote.most_common(1)[0][0]
            return {
                "answer": label,
                "method": "FAISS 만장일치",
                "references": [(doc, score) for doc, score in filtered],
            }

    # 불확실: Agent에게 판단 위임, 실패 시 FAISS 다수결 fallback
    try:
        result = agent.invoke(
            {"messages": [("human", f"경보 메시지 '{alarm_message}'의 장애 유형을 분류해주세요.")]},
            {"recursion_limit": 6},
        )
        return {
            "answer": result["messages"][-1].content,
            "method": "Agent 판단",
            "messages": result["messages"],
        }
    except Exception:
        # Agent 실패 시 FAISS k=3 다수결로 fallback (threshold 1.2로 완화)
        results = vectorstore.similarity_search_with_score(alarm_message, k=3)
        fallback = [(doc, score) for doc, score in results if score < 1.2]
        if fallback:
            labels = [doc.metadata.get("label") for doc, _ in fallback]
            label = Counter(labels).most_common(1)[0][0]
        else:
            label = "Unknown"
        return {
            "answer": label,
            "method": "FAISS fallback",
            "references": fallback,
        }


def load_agent(_vectorstore):

    @tool
    def search_similar_alarms(alarm_message: str) -> str:
        """유사 경보 검색"""
        results = _vectorstore.similarity_search_with_score(alarm_message, k=3)

        threshold = 0.8
        filtered = [(doc, score) for doc, score in results if score < threshold]

        if not filtered:
            return "유사한 경보 사례를 찾지 못했습니다. refine_search 도구를 사용해 재검색하세요."

        result_text = ""
        for doc, score in filtered:
            label = doc.metadata.get("label", "Unknown")
            result_text += f"경보: {doc.page_content} | 장애유형: {label} (거리: {score:.4f})\n"

        return result_text

    @tool
    def refine_search(alarm_message: str) -> str:
        """재검색"""
        keywords = alarm_message.replace("_", " ").replace("-", " ").replace("(", " ").replace(")", " ")
        keywords = " ".join(keywords.split()[:3])

        results = _vectorstore.similarity_search_with_score(keywords, k=3)

        threshold = 1.2
        filtered = [(doc, score) for doc, score in results if score < threshold]

        if not filtered:
            return "재검색에서도 유사한 사례를 찾지 못했습니다. 알려진 패턴이 없는 경보입니다."

        result_text = f"[재검색 키워드: {keywords}]\n"
        for doc, score in filtered:
            label = doc.metadata.get("label", "Unknown")
            result_text += f"경보: {doc.page_content} | 장애유형: {label} (거리: {score:.4f})\n"

        return result_text

    llm = ChatGroq(model="llama-3.1-8b-instant")
    return create_react_agent(
        llm,
        [search_similar_alarms, refine_search],
        prompt="""You are a network fault classification expert.

STEP 1: Call search_similar_alarms ONCE with the alarm message.
STEP 2: If no results found, call refine_search ONCE.
STEP 3: Based on the tool results, reply with ONLY a plain text message.

IMPORTANT RULES:
- Call each tool AT MOST ONCE. Never repeat the same tool call.
- You may ONLY use these tools: search_similar_alarms, refine_search
- Do NOT call any other tool. Do NOT create new tools.
- Your FINAL answer must be a plain text message (NOT a tool call).
- Your FINAL answer must be exactly one word: LinkCut, PowerFail, or UnitFail.
- Do not explain. Do not add any other text."""
    )
