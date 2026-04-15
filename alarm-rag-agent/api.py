"""
FastAPI 서버 — 통신 경보 장애 분류 API

실행:
    uvicorn api:app --reload --port 8000

확인:
    브라우저에서 http://localhost:8000/docs (Swagger UI 자동 생성)
    curl -X POST http://localhost:8000/classify \
        -H "Content-Type: application/json" \
        -d '{"alarm_message": "BATTERY_FAIL"}'

설계 메모
---------
- Streamlit UI와 동일한 LangGraph를 재사용한다 (agent.build_graph).
- 벡터스토어·그래프 로드는 무겁기 때문에 lifespan에서 단 한 번만 수행.
- 요청/응답 스키마를 Pydantic으로 명시해 백엔드 개발자가 계약을 바로 읽을 수 있도록 함.
- 매 요청마다 구조화된 JSON 로그 1줄을 stdout에 남긴다
  (alarm, method, latency_ms, cached, tool_calls) — 운영 지표 수집의 기본.
- 동일 경보 반복 입력은 in-memory dict 캐시로 응답 (무상태 단일 프로세스 기준).
"""

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from agent import build_graph, classify_alarm, load_vectorstore


# ---------- 로깅 ----------

_logger = logging.getLogger("alarm_api")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False


def _log_event(event: str, **fields) -> None:
    """stdout에 한 줄짜리 JSON 로그를 남긴다. 파싱·수집 용이."""
    payload = {"event": event, **fields}
    _logger.info(json.dumps(payload, ensure_ascii=False))


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
    latency_ms: float = Field(
        ..., description="분류 소요시간(ms). 캐시 히트 시 0에 가깝다."
    )
    cached: bool = Field(
        False, description="동일 경보가 캐시에 있어 재계산 없이 응답한 경우 True"
    )


# ---------- 라이프사이클 ----------

_state: dict = {}
_cache: dict = {}  # alarm_message -> ClassifyResponse dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 무거운 리소스는 한 번만 로드
    _log_event("startup_begin")
    t0 = time.perf_counter()
    vectorstore = load_vectorstore()
    _state["graph"] = build_graph(vectorstore)
    _log_event(
        "startup_done",
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    yield
    _state.clear()
    _cache.clear()


# ---------- 앱 ----------

app = FastAPI(
    title="통신 경보 장애 분류 API",
    description=(
        "경보 메시지를 받아 LinkCut / PowerFail / UnitFail 중 하나로 분류하는 RAG Agent. "
        "FAISS 만장일치 → LLM ReAct Agent → FAISS 다수결 fallback 3단 구조를 "
        "LangGraph StateGraph로 구현."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """헬스체크 — 서빙 중인지만 확인."""
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """간단한 운영 지표 — 캐시 항목 수."""
    return {"cache_size": len(_cache)}


def _count_tool_calls(messages) -> int:
    if not messages:
        return 0
    return sum(1 for m in messages if isinstance(m, ToolMessage))


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """경보 메시지를 분류한다."""
    graph = _state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    alarm = req.alarm_message.strip()

    cached = _cache.get(alarm)
    if cached is not None:
        _log_event(
            "classify",
            alarm=alarm,
            method=cached["method"],
            latency_ms=0.0,
            cached=True,
            tool_calls=cached.get("tool_calls", 0),
        )
        return ClassifyResponse(**cached)

    t0 = time.perf_counter()
    result = classify_alarm(graph, alarm)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

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

    tool_calls = _count_tool_calls(result.get("messages"))

    response = ClassifyResponse(
        classification=result["answer"] or "Unknown",
        method=result["method"] or "Unknown",
        references=refs,
        latency_ms=latency_ms,
        cached=False,
    )

    # 캐시에는 ReferenceItem을 dict로 직렬화해 저장 (TTL 없이 단순 LRU 느낌)
    _cache[alarm] = {
        "classification": response.classification,
        "method": response.method,
        "references": [r.model_dump() for r in refs] if refs else None,
        "latency_ms": latency_ms,
        "cached": True,
        "tool_calls": tool_calls,
    }
    # 단순 상한 — 1000건 넘으면 오래된 것부터 버림 (Python 3.7+ dict 입력순서 보장)
    if len(_cache) > 1000:
        for k in list(_cache.keys())[:100]:
            _cache.pop(k, None)

    _log_event(
        "classify",
        alarm=alarm,
        method=response.method,
        latency_ms=latency_ms,
        cached=False,
        tool_calls=tool_calls,
    )
    return response
