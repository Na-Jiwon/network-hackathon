"""
classify_alarm 3분기 테스트

LLM·HuggingFace 임베딩을 타지 않도록 벡터스토어·Agent를 전부 모킹한다.
- FAISS 만장일치 경로 : similarity_search_with_score가 같은 라벨 3건을 반환
- Agent 판단 경로     : FAISS 결과가 혼재 → agent.invoke 호출되어 라벨 반환
- FAISS fallback 경로 : FAISS 결과 혼재 + agent.invoke가 예외 → threshold 1.2 다수결
"""

from collections import namedtuple
from unittest.mock import MagicMock

import pytest

from agent import build_graph, classify_alarm


FakeDoc = namedtuple("FakeDoc", ["page_content", "metadata"])


def _doc(text: str, label: str) -> FakeDoc:
    return FakeDoc(page_content=text, metadata={"label": label})


def _fake_vectorstore(results_by_call):
    """`similarity_search_with_score` 호출 순서대로 결과를 뱉는 가짜 벡터스토어."""
    vs = MagicMock()
    vs.similarity_search_with_score.side_effect = list(results_by_call)
    return vs


def _fake_agent(final_text: str = "LinkCut", raise_exc: bool = False):
    agent = MagicMock()
    if raise_exc:
        agent.invoke.side_effect = RuntimeError("simulated agent failure")
    else:
        last = MagicMock()
        last.content = final_text
        agent.invoke.return_value = {"messages": [last]}
    return agent


# ---------- 1. FAISS 만장일치 ----------

def test_faiss_unanimous_path():
    results = [
        (_doc("ETH-ERR-1", "LinkCut"), 0.5),
        (_doc("ETH-ERR-2", "LinkCut"), 0.55),
        (_doc("ETH-ERR-3", "LinkCut"), 0.6),
    ]
    vs = _fake_vectorstore([results])
    agent = _fake_agent()

    graph = build_graph(vs, agent=agent)
    out = classify_alarm(graph, "ETH-ERR")

    assert out["answer"] == "LinkCut"
    assert out["method"] == "FAISS 만장일치"
    assert len(out["references"]) == 3
    # Agent는 호출되지 않아야 함
    agent.invoke.assert_not_called()


# ---------- 2. Agent 판단 ----------

def test_agent_path_when_labels_mixed():
    # 라벨이 섞여 만장일치가 아님 → agent_refine 노드로 분기
    results = [
        (_doc("ETH-ERR", "LinkCut"), 0.5),
        (_doc("PSU-FAIL", "PowerFail"), 0.6),
        (_doc("BOARD-ERR", "UnitFail"), 0.7),
    ]
    vs = _fake_vectorstore([results])
    agent = _fake_agent(final_text="PowerFail")

    graph = build_graph(vs, agent=agent)
    out = classify_alarm(graph, "혼재된 경보")

    assert "PowerFail" in (out["answer"] or "")
    assert out["method"] == "Agent 판단"
    agent.invoke.assert_called_once()


# ---------- 3. FAISS fallback (Agent 예외) ----------

def test_fallback_path_when_agent_raises():
    # 1차(k=3) : 라벨 혼재  → agent로 분기
    # 2차(k=3, fallback) : threshold 1.2 이하 다수결 → PowerFail 2표
    first = [
        (_doc("ETH-ERR", "LinkCut"), 0.5),
        (_doc("PSU-FAIL", "PowerFail"), 0.6),
        (_doc("BOARD-ERR", "UnitFail"), 0.7),
    ]
    fallback = [
        (_doc("PSU-FAIL", "PowerFail"), 0.9),
        (_doc("BATTERY_FAIL", "PowerFail"), 1.1),
        (_doc("ETH-ERR", "LinkCut"), 1.0),
    ]
    vs = _fake_vectorstore([first, fallback])
    agent = _fake_agent(raise_exc=True)

    graph = build_graph(vs, agent=agent)
    out = classify_alarm(graph, "?")

    assert out["method"] == "FAISS fallback"
    assert out["answer"] == "PowerFail"
    assert len(out["references"]) == 3


# ---------- 4. 임계값 초과 시 Agent 경로 ----------

def test_agent_path_when_all_above_threshold():
    # 모든 거리가 0.8 이상 → filtered가 비어 unclear → agent로 분기
    results = [
        (_doc("X", "LinkCut"), 0.85),
        (_doc("Y", "PowerFail"), 0.9),
        (_doc("Z", "UnitFail"), 0.95),
    ]
    vs = _fake_vectorstore([results])
    agent = _fake_agent(final_text="UnitFail")

    graph = build_graph(vs, agent=agent)
    out = classify_alarm(graph, "미지의 경보")

    assert "UnitFail" in (out["answer"] or "")
    assert out["method"] == "Agent 판단"
    agent.invoke.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
