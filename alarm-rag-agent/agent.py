"""
경보 분류 그래프 정의

3-branch 분류 로직을 LangGraph StateGraph의 노드/조건부 엣지로 구성한다.

    fast_path_faiss
        └── [route_by_confidence]
              ├── unanimous   → finalize_unanimous → END
              └── unclear     → agent_refine
                                  └── [route_after_agent]
                                        ├── done     → END
                                        └── fallback → fallback_vote → END

각 노드는 state에 중간 근거(references, messages)를 누적하므로, API/Streamlit UI
양쪽에서 동일 state를 받아 분기별 근거 표시를 그대로 재사용할 수 있다.
"""

import os
from collections import Counter
from typing import Any, List, Optional, Tuple, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

load_dotenv()


# ---------- 벡터스토어 ----------

def load_vectorstore():
    """기존 인덱스 로드, 없으면 학습 CSV로 생성.

    실 데이터(`data/Q2_train.csv`)가 없을 때는 구조 확인용 더미 샘플
    (`data/sample_Q2_train.csv`)로 fallback한다. 외부 리뷰어가 실 데이터
    없이도 파이프라인이 끝까지 도는지 바로 확인할 수 있도록 하기 위함.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    if os.path.exists("faiss_index"):
        return FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    csv_path = "data/Q2_train.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/sample_Q2_train.csv"
    df = pd.read_csv(csv_path)
    df_unique = df[["alarmmsg_original", "root_cause_type"]].drop_duplicates()
    docs = df_unique["alarmmsg_original"].tolist()
    labels = df_unique["root_cause_type"].tolist()
    vectorstore = FAISS.from_texts(
        docs, embeddings, metadatas=[{"label": l} for l in labels]
    )
    vectorstore.save_local("faiss_index")
    return vectorstore


# ---------- ReAct Agent (그래프의 agent_refine 노드에서 사용) ----------

def load_agent(vectorstore):
    """refine_search / search_similar_alarms 2개 도구를 쓰는 ReAct Agent."""

    @tool
    def search_similar_alarms(alarm_message: str) -> str:
        """유사 경보 검색 (1차, threshold 0.8)"""
        results = vectorstore.similarity_search_with_score(alarm_message, k=3)
        filtered = [(doc, score) for doc, score in results if score < 0.8]
        if not filtered:
            return "유사한 경보 사례를 찾지 못했습니다. refine_search 도구를 사용해 재검색하세요."
        lines = []
        for doc, score in filtered:
            label = doc.metadata.get("label", "Unknown")
            lines.append(
                f"경보: {doc.page_content} | 장애유형: {label} (거리: {score:.4f})"
            )
        return "\n".join(lines)

    @tool
    def refine_search(alarm_message: str) -> str:
        """재검색 (threshold 1.2로 완화, 첫 3개 토큰만 추출)"""
        keywords = (
            alarm_message.replace("_", " ")
            .replace("-", " ")
            .replace("(", " ")
            .replace(")", " ")
        )
        keywords = " ".join(keywords.split()[:3])
        results = vectorstore.similarity_search_with_score(keywords, k=3)
        filtered = [(doc, score) for doc, score in results if score < 1.2]
        if not filtered:
            return "재검색에서도 유사한 사례를 찾지 못했습니다. 알려진 패턴이 없는 경보입니다."
        lines = [f"[재검색 키워드: {keywords}]"]
        for doc, score in filtered:
            label = doc.metadata.get("label", "Unknown")
            lines.append(
                f"경보: {doc.page_content} | 장애유형: {label} (거리: {score:.4f})"
            )
        return "\n".join(lines)

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
- Do not explain. Do not add any other text.""",
    )


# ---------- LangGraph State ----------

class ClassifyState(TypedDict, total=False):
    alarm_message: str
    answer: Optional[str]
    method: Optional[str]
    references: List[Tuple[Any, float]]
    messages: List[Any]
    agent_failed: bool


# ---------- StateGraph ----------

def build_graph(vectorstore, agent=None):
    """3-branch 분류 로직을 LangGraph StateGraph로 구성.

    Parameters
    ----------
    vectorstore : FAISS
    agent : optional
        미리 만들어둔 ReAct agent. None이면 `load_agent(vectorstore)` 로 생성.
    """
    if agent is None:
        agent = load_agent(vectorstore)

    # --- 노드 ---

    def fast_path_faiss(state: ClassifyState) -> ClassifyState:
        """1차 FAISS 검색 (threshold 0.8). 결과를 state.references에 저장."""
        alarm = state["alarm_message"]
        results = vectorstore.similarity_search_with_score(alarm, k=3)
        filtered = [(doc, score) for doc, score in results if score < 0.8]
        return {"references": filtered}

    def finalize_unanimous(state: ClassifyState) -> ClassifyState:
        """k=3 만장일치: FAISS 결과를 그대로 신뢰."""
        filtered = state["references"]
        label = filtered[0][0].metadata.get("label")
        return {"answer": label, "method": "FAISS 만장일치"}

    def agent_refine(state: ClassifyState) -> ClassifyState:
        """불확실 경로: ReAct Agent에 위임. 실패 시 agent_failed=True."""
        alarm = state["alarm_message"]
        try:
            result = agent.invoke(
                {
                    "messages": [
                        (
                            "human",
                            f"경보 메시지 '{alarm}'의 장애 유형을 분류해주세요.",
                        )
                    ]
                },
                {"recursion_limit": 6},
            )
            return {
                "answer": result["messages"][-1].content,
                "method": "Agent 판단",
                "messages": result["messages"],
                "agent_failed": False,
            }
        except Exception:
            return {"agent_failed": True}

    def fallback_vote(state: ClassifyState) -> ClassifyState:
        """Agent 실패 시 FAISS k=3 다수결 (threshold 1.2로 완화)."""
        alarm = state["alarm_message"]
        results = vectorstore.similarity_search_with_score(alarm, k=3)
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

    # --- 조건부 엣지 라우터 ---

    def route_by_confidence(state: ClassifyState) -> str:
        filtered = state.get("references") or []
        if filtered:
            labels = [doc.metadata.get("label") for doc, _ in filtered]
            if len(set(labels)) == 1:
                return "unanimous"
        return "unclear"

    def route_after_agent(state: ClassifyState) -> str:
        return "fallback" if state.get("agent_failed") else "done"

    # --- 그래프 조립 ---

    graph = StateGraph(ClassifyState)
    graph.add_node("fast_path_faiss", fast_path_faiss)
    graph.add_node("finalize_unanimous", finalize_unanimous)
    graph.add_node("agent_refine", agent_refine)
    graph.add_node("fallback_vote", fallback_vote)

    graph.set_entry_point("fast_path_faiss")
    graph.add_conditional_edges(
        "fast_path_faiss",
        route_by_confidence,
        {"unanimous": "finalize_unanimous", "unclear": "agent_refine"},
    )
    graph.add_edge("finalize_unanimous", END)
    graph.add_conditional_edges(
        "agent_refine",
        route_after_agent,
        {"done": END, "fallback": "fallback_vote"},
    )
    graph.add_edge("fallback_vote", END)

    return graph.compile()


# ---------- 공개 분류 API ----------

VALID_LABELS = ("LinkCut", "PowerFail", "UnitFail")


def normalize_label(text) -> str:
    """LLM 원문이든 메타데이터 라벨이든 세 라벨 중 하나 또는 'Unknown'으로 정규화.

    Agent 경로에서 LLM이 "The answer is LinkCut" 같이 답하는 경우가 있어
    API/UI에서 그대로 노출하지 않도록 단일 지점에서 정규화한다.
    """
    if not text:
        return "Unknown"
    text = str(text)
    for label in VALID_LABELS:
        if label in text:
            return label
    return "Unknown"


def classify_alarm(graph, alarm_message: str) -> dict:
    """그래프 한 번 실행해 단건 경보를 분류한다.

    반환 키:
        answer       : 'LinkCut' / 'PowerFail' / 'UnitFail' / 'Unknown' 중 하나 (정규화됨)
        method       : 'FAISS 만장일치' | 'Agent 판단' | 'FAISS fallback'
        references   : [(Document, distance), ...]  (FAISS 경로에서)
        messages     : ReAct Agent 실행 메시지들  (Agent 경로에서)
    """
    result = graph.invoke({"alarm_message": alarm_message})
    return {
        "answer": normalize_label(result.get("answer")),
        "method": result.get("method"),
        "references": result.get("references"),
        "messages": result.get("messages"),
    }
