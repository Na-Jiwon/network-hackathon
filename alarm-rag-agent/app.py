import streamlit as st
from langchain_core.messages import ToolMessage

from agent import build_graph, classify_alarm, load_vectorstore


def get_confidence(score):
    """L2 거리 기반 confidence 수준 반환 (explore_data.py 분석 근거)"""
    if score < 0.65:
        return "🟢 높은 확신", "success"
    elif score < 0.90:
        return "🟡 보통", "warning"
    else:
        return "🔴 낮은 확신 — 수동 확인 권장", "error"


def extract_fault_type(text):
    """텍스트에서 장애유형 추출"""
    for label in ["LinkCut", "PowerFail", "UnitFail"]:
        if label in text:
            return label
    return None


def extract_tool_results(messages):
    """Agent 응답에서 도구 호출 결과 추출"""
    tool_results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results.append({
                "tool": msg.name,
                "content": msg.content,
            })
    return tool_results


@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


@st.cache_resource
def get_graph(_vectorstore):
    return build_graph(_vectorstore)


st.title("🔔 통신 경보 장애 분류 Agent")
st.write("경보 메시지를 입력하면 AI가 장애 유형을 자동으로 분류합니다.")

with st.spinner("AI 준비 중..."):
    vectorstore = get_vectorstore()
    graph = get_graph(vectorstore)

alarm_input = st.text_input(
    "경보 메시지 입력",
    placeholder="예) ETH-ERR, PSU-FAIL, BATTERY_FAIL"
)

if st.button("분류하기"):
    if alarm_input:
        with st.spinner("분석 중..."):
            result = classify_alarm(graph, alarm_input)
            answer = result["answer"] or "Unknown"
            method = result["method"]

        st.success("분류 완료!")
        classified_type = extract_fault_type(answer)

        st.write("### 결과")
        st.write(classified_type or answer)

        if method == "FAISS 만장일치":
            refs = result["references"]
            best_score = refs[0][1]
            confidence_label, confidence_type = get_confidence(best_score)

            st.write("### 분류 신뢰도")
            getattr(st, confidence_type)(
                f"{confidence_label} (최근접 거리: {best_score:.4f}, 낮을수록 유사)"
            )

            st.write("### 분류 근거")
            st.caption("📌 FAISS 만장일치 — k=3 검색 결과가 동일 장애유형")
            for i, (doc, score) in enumerate(refs, 1):
                label = doc.metadata.get("label")
                st.markdown(
                    f":green[{i}. 경보: {doc.page_content} | 장애유형: {label} | 거리: {score:.4f} (낮을수록 유사) ✅]"
                )

        elif method == "FAISS fallback":
            # Agent 실패, FAISS 다수결 fallback
            refs = result.get("references") or []
            if refs:
                best_score = refs[0][1]
                confidence_label, confidence_type = get_confidence(best_score)
                st.write("### 분류 신뢰도")
                getattr(st, confidence_type)(
                    f"{confidence_label} (최근접 거리: {best_score:.4f}, 낮을수록 유사)"
                )

            st.write("### 분류 근거")
            st.caption("📌 FAISS 다수결 — Agent 오류로 FAISS k=3 다수결 분류")
            for i, (doc, score) in enumerate(refs, 1):
                label = doc.metadata.get("label")
                match = label == classified_type
                if match:
                    st.markdown(
                        f":green[{i}. 경보: {doc.page_content} | 장애유형: {label} | 거리: {score:.4f} (낮을수록 유사) ✅]"
                    )
                else:
                    st.markdown(
                        f":gray[{i}. 경보: {doc.page_content} | 장애유형: {label} | 거리: {score:.4f} (낮을수록 유사) ❌]"
                    )

        else:
            # Agent 판단: 도구 호출 결과 표시
            results_with_score = vectorstore.similarity_search_with_score(alarm_input, k=1)
            if results_with_score:
                best_score = results_with_score[0][1]
                confidence_label, confidence_type = get_confidence(best_score)
                st.write("### 분류 신뢰도")
                getattr(st, confidence_type)(
                    f"{confidence_label} (최근접 거리: {best_score:.4f}, 낮을수록 유사)"
                )

            st.write("### 분류 근거")
            st.caption("🤖 Agent 판단 — FAISS 결과가 불확실하여 LLM이 최종 분류")
            messages = result.get("messages") or []
            tool_results = extract_tool_results(messages)

            if tool_results:
                for tr in tool_results:
                    tool_label = "1차 검색" if tr["tool"] == "search_similar_alarms" else "재검색 (refine)"
                    st.caption(f"🔧 {tool_label}")

                    lines = [l.strip() for l in tr["content"].strip().split("\n") if l.strip()]
                    for line in lines:
                        case_type = extract_fault_type(line)
                        match = case_type == classified_type if classified_type and case_type else None

                        if match is True:
                            st.markdown(f":green[{line} ✅ 일치]")
                        elif match is False:
                            st.markdown(f":gray[{line} ❌ 불일치]")
                        else:
                            st.write(line)
            else:
                st.info("Agent가 도구를 호출하지 않았습니다.")
    else:
        st.warning("경보 메시지를 입력해주세요.")
