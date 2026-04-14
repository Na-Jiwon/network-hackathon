"""
FastAPI 서버 — 통신 경보 장애 분류 API

실행:
    uvicorn api:app --reload --port 8000

확인:
    브라우저에서 http://localhost:8000/docs (Swagger UI 자동 생성)
    curl -X POST http://localhost:8000/classify \
        -H "Content-Type: application/json" \
        -d '{"alarm_message": "BATTERY_FAIL"}'

설계 메모:
- Streamlit UI와 동일한 LangGraph를 재사용한다 (agent.build_graph).
- 벡터스토어·그래프 로드는 무겁기 때문에 lifespan에서 단 한 번만 수행.
- 요청/응답 스키마를 Pydantic으로 명시해 백엔드 개발자가 계약을 바로 읽을 수 있도록 함.
"""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import build_graph, classify_alarm, load_vectorstore


# ---------- 스키마 ----------

class ClassifyRequest(BaseModel):
    alarm_message: str = Field(
        ..., min_length=1, description="분류 대상 경보 메시지 (예: BATTERY_FAIL)"
    )


class ReferenceItem(BaseModel):
    alarm: str = Field(..., description="유사 경보 원문")
    label: str = Field(..., description="해당 경보의 장애유형")
    distance: float = Field(..., description="L2 거리 (낮을수록 유사)")


class ClassifyResponse(BaseModel):
    classification: str = Field(
        ..., description="LinkCut / PowerFail / UnitFail / Unknown 중 하나"
    )
    method: str = Field(
        ..., description="분류 경로: 'FAISS 만장일치' | 'Agent 판단' | 'FAISS fallback'"
    )
    references: Optional[List[ReferenceItem]] = Field(
        None, description="FAISS 경로일 때 참고된 유사 사례 (Agent 경로에서는 null)"
    )


# ---------- 라이프사이클 ----------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 무거운 리소스는 한 번만 로드
    vectorstore = load_vectorstore()
    _state["graph"] = build_graph(vectorstore)
    yield
    _state.clear()


# ---------- 앱 ----------

app = FastAPI(
    title="통신 경보 장애 분류 API",
    description=(
        "경보 메시지를 받아 LinkCut / PowerFail / UnitFail 중 하나로 분류하는 RAG Agent. "
        "FAISS 만장일치 → LLM ReAct Agent → FAISS 다수결 fallback 3단 구조."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """헬스체크 — 서빙 중인지만 확인."""
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """경보 메시지를 분류한다."""
    graph = _state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    result = classify_alarm(graph, req.alarm_message)

    refs: Optional[List[ReferenceItem]] = None
    if result.get("references"):
        refs = [
            ReferenceItem(
                alarm=doc.page_content,
                label=doc.metadata.get("label", "Unknown"),
                distance=float(score),
            )
            for doc, score in result["references"]
        ]

    return ClassifyResponse(
        classification=result["answer"] or "Unknown",
        method=result["method"] or "Unknown",
        references=refs,
    )
