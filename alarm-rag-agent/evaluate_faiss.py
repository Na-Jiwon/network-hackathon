import pandas as pd
import numpy as np
import time
import sys
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from collections import Counter
from agent import build_graph, classify_alarm, load_agent

load_dotenv()

# 샘플 수 설정 (커맨드라인 인자로 변경 가능: python evaluate_faiss.py 100)
LLM_SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 30
FAISS_SAMPLE_SIZE = 500
RETRY_WAIT = 10  # rate limit 시 대기 시간(초)
MAX_RETRIES = 3

# Groq llama-3.1-8b-instant 가격 (2025-04 기준, $/1M tokens)
# 공식 가격표: https://groq.com/pricing  (변경 시 여기만 수정)
GROQ_PRICE_INPUT_PER_1M = 0.05
GROQ_PRICE_OUTPUT_PER_1M = 0.08


def percentile(values, p):
    """numpy 없이 p50/p95 계산 (values는 ms 단위 리스트)"""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def usage_from_response(response):
    """ChatGroq 응답에서 input/output 토큰 추출 (없으면 0)"""
    meta = getattr(response, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0) or 0), int(meta.get("output_tokens", 0) or 0)


def usage_from_agent_messages(messages):
    """Agent 응답 메시지들에서 누적 토큰 추출"""
    in_tok, out_tok = 0, 0
    for m in messages:
        meta = getattr(m, "usage_metadata", None) or {}
        in_tok += int(meta.get("input_tokens", 0) or 0)
        out_tok += int(meta.get("output_tokens", 0) or 0)
    return in_tok, out_tok


def estimate_cost(total_in, total_out, n_samples):
    """1000건 기준 예상 비용 계산 ($)"""
    if n_samples == 0:
        return 0.0
    per_call_in = total_in / n_samples
    per_call_out = total_out / n_samples
    cost_1k = (
        per_call_in * 1000 * GROQ_PRICE_INPUT_PER_1M / 1_000_000
        + per_call_out * 1000 * GROQ_PRICE_OUTPUT_PER_1M / 1_000_000
    )
    return cost_1k

df = pd.read_csv("data/Q2_train.csv")

# ticketno 단위 split (leakage 방지)
tickets = df['ticketno'].unique()
train_tickets, test_tickets = train_test_split(
    tickets, test_size=0.2, random_state=42
)
train_df = df[df['ticketno'].isin(train_tickets)]
test_df = df[df['ticketno'].isin(test_tickets)]

print(f"Train: {len(train_df)}개 / Test: {len(test_df)}개")
print(f"FAISS 평가: {FAISS_SAMPLE_SIZE}개 / LLM 평가: {LLM_SAMPLE_SIZE}개")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

if os.path.exists("faiss_eval_index"):
    vectorstore = FAISS.load_local(
        "faiss_eval_index", embeddings, allow_dangerous_deserialization=True
    )
else:
    train_unique = train_df[['alarmmsg_original', 'root_cause_type']].drop_duplicates()
    docs = []
    labels = []
    for _, row in train_unique.iterrows():
        docs.append(row['alarmmsg_original'])
        labels.append(row['root_cause_type'])
    vectorstore = FAISS.from_texts(
        docs, embeddings,
        metadatas=[{"label": l} for l in labels]
    )
    vectorstore.save_local("faiss_eval_index")


def extract_label(text):
    for label in ["LinkCut", "PowerFail", "UnitFail"]:
        if label in text:
            return label
    return "Unknown"


def invoke_with_retry(func, max_retries=MAX_RETRIES):
    """Groq rate limit 대응: 실패 시 대기 후 재시도"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait = RETRY_WAIT * (attempt + 1)
                print(f"    ⏳ rate limit 도달, {wait}초 대기 중...")
                time.sleep(wait)
            else:
                raise e
    raise Exception(f"{max_retries}회 재시도 실패")


# === FAISS 단독 평가 ===
print(f"\n=== FAISS 단독 평가 ({FAISS_SAMPLE_SIZE}개) ===")
faiss_sample = test_df.sample(min(FAISS_SAMPLE_SIZE, len(test_df)), random_state=42)
y_true, y_pred_faiss = [], []
faiss_latencies = []  # ms, 단건 분류 latency

for _, row in faiss_sample.iterrows():
    t0 = time.perf_counter()
    results = vectorstore.similarity_search(row['alarmmsg_original'], k=3)
    labels = [r.metadata['label'] for r in results]
    pred = Counter(labels).most_common(1)[0][0]
    faiss_latencies.append((time.perf_counter() - t0) * 1000)
    y_true.append(row['root_cause_type'])
    y_pred_faiss.append(pred)

print(f"Accuracy: {accuracy_score(y_true, y_pred_faiss):.4f}")
print(
    f"Latency (단건 분류): p50 {percentile(faiss_latencies, 50):.1f}ms / "
    f"p95 {percentile(faiss_latencies, 95):.1f}ms"
)
print(classification_report(y_true, y_pred_faiss))


# === RAG (LLM) 평가 ===
print(f"\n=== RAG Agent 평가 ({LLM_SAMPLE_SIZE}개) ===")
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_sample = test_df.sample(min(LLM_SAMPLE_SIZE, len(test_df)), random_state=42)
y_true_llm = [row['root_cause_type'] for _, row in llm_sample.iterrows()]
y_pred_rag = []
rag_latencies = []
rag_in_tokens, rag_out_tokens = 0, 0

for i, (_, row) in enumerate(llm_sample.iterrows()):
    t0 = time.perf_counter()
    results = vectorstore.similarity_search_with_score(row['alarmmsg_original'], k=3)
    filtered = [(r, s) for r, s in results if s < 0.8]
    if not filtered:
        results = vectorstore.similarity_search_with_score(row['alarmmsg_original'], k=3)
        filtered = [(r, s) for r, s in results if s < 1.2]
    context = "\n".join([
        f"경보: {r.page_content} | 장애유형: {r.metadata['label']}"
        for r, _ in filtered
    ])
    prompt = f"""유사한 과거 경보 사례:
{context}

새로운 경보: {row['alarmmsg_original']}
LinkCut, PowerFail, UnitFail 중 하나만 답하세요."""

    try:
        response = invoke_with_retry(lambda: llm.invoke(prompt))
        pred = extract_label(response.content)
        in_tok, out_tok = usage_from_response(response)
        rag_in_tokens += in_tok
        rag_out_tokens += out_tok
    except Exception as e:
        print(f"  {i+1} 오류: {e}")
        pred = "Unknown"
    rag_latencies.append((time.perf_counter() - t0) * 1000)
    y_pred_rag.append(pred)
    print(f"  {i+1}/{LLM_SAMPLE_SIZE} 완료")

print(f"Accuracy: {accuracy_score(y_true_llm, y_pred_rag):.4f}")
print(
    f"Latency (단건 분류): p50 {percentile(rag_latencies, 50):.1f}ms / "
    f"p95 {percentile(rag_latencies, 95):.1f}ms"
)
rag_cost_1k = estimate_cost(rag_in_tokens, rag_out_tokens, LLM_SAMPLE_SIZE)
print(
    f"Groq 토큰: in {rag_in_tokens} / out {rag_out_tokens} "
    f"(1000건당 예상 비용 ${rag_cost_1k:.4f})"
)
print(classification_report(y_true_llm, y_pred_rag, zero_division=0))


# === Agent 평가 ===
print(f"\n=== Agent 평가 (refine_search 포함, {LLM_SAMPLE_SIZE}개) ===")
agent = load_agent(vectorstore)
y_pred_agent = []
agent_latencies = []
agent_in_tokens, agent_out_tokens = 0, 0

for i, (_, row) in enumerate(llm_sample.iterrows()):
    t0 = time.perf_counter()
    try:
        result = invoke_with_retry(lambda r=row: agent.invoke({
            "messages": [("human", f"경보 메시지 '{r['alarmmsg_original']}'의 장애 유형을 분류해주세요.")]
        }))
        pred = extract_label(result["messages"][-1].content)
        in_tok, out_tok = usage_from_agent_messages(result["messages"])
        agent_in_tokens += in_tok
        agent_out_tokens += out_tok
    except Exception as e:
        print(f"  {i+1} 오류: {e}")
        pred = "Unknown"
    agent_latencies.append((time.perf_counter() - t0) * 1000)
    y_pred_agent.append(pred)
    print(f"  {i+1}/{LLM_SAMPLE_SIZE} 완료")

print(f"Accuracy: {accuracy_score(y_true_llm, y_pred_agent):.4f}")
print(
    f"Latency (단건 분류): p50 {percentile(agent_latencies, 50):.1f}ms / "
    f"p95 {percentile(agent_latencies, 95):.1f}ms"
)
agent_cost_1k = estimate_cost(agent_in_tokens, agent_out_tokens, LLM_SAMPLE_SIZE)
print(
    f"Groq 토큰: in {agent_in_tokens} / out {agent_out_tokens} "
    f"(1000건당 예상 비용 ${agent_cost_1k:.4f})"
)
print(classification_report(y_true_llm, y_pred_agent, zero_division=0))


# === 하이브리드 평가 ===
print(f"\n=== 하이브리드 평가 (LangGraph StateGraph, {LLM_SAMPLE_SIZE}개) ===")
graph = build_graph(vectorstore, agent=agent)
y_pred_hybrid = []
faiss_count = 0
agent_count = 0
hybrid_latencies = []
hybrid_in_tokens, hybrid_out_tokens = 0, 0

for i, (_, row) in enumerate(llm_sample.iterrows()):
    t0 = time.perf_counter()
    try:
        result = invoke_with_retry(
            lambda r=row: classify_alarm(graph, r['alarmmsg_original'])
        )
        pred = extract_label(result["answer"])
        if result["method"] == "FAISS 만장일치":
            faiss_count += 1
        else:
            agent_count += 1
            # Agent 경로만 토큰 집계 (FAISS 경로는 LLM 호출 없음)
            in_tok, out_tok = usage_from_agent_messages(result.get("messages", []))
            hybrid_in_tokens += in_tok
            hybrid_out_tokens += out_tok
    except Exception as e:
        print(f"  {i+1} 오류: {e}")
        pred = "Unknown"
    hybrid_latencies.append((time.perf_counter() - t0) * 1000)
    y_pred_hybrid.append(pred)
    print(f"  {i+1}/{LLM_SAMPLE_SIZE} 완료")

print(f"Accuracy: {accuracy_score(y_true_llm, y_pred_hybrid):.4f}")
print(f"FAISS 만장일치: {faiss_count}건 / Agent 판단: {agent_count}건")
print(
    f"Latency (단건 분류): p50 {percentile(hybrid_latencies, 50):.1f}ms / "
    f"p95 {percentile(hybrid_latencies, 95):.1f}ms"
)
hybrid_cost_1k = estimate_cost(hybrid_in_tokens, hybrid_out_tokens, LLM_SAMPLE_SIZE)
print(
    f"Groq 토큰: in {hybrid_in_tokens} / out {hybrid_out_tokens} "
    f"(1000건당 예상 비용 ${hybrid_cost_1k:.4f}, FAISS 경로는 LLM 호출 없음)"
)
print(classification_report(y_true_llm, y_pred_hybrid, zero_division=0))


# === 최종 비교 ===
print(f"\n=== 최종 비교 ===")
print(f"FAISS 단독:  {accuracy_score(y_true, y_pred_faiss):.4f} (단건 경보 {len(faiss_sample)}개 샘플)")
print(f"RAG (LLM):   {accuracy_score(y_true_llm, y_pred_rag):.4f} (단건 경보 {LLM_SAMPLE_SIZE}개 샘플)")
print(f"Agent:       {accuracy_score(y_true_llm, y_pred_agent):.4f} (단건 경보 {LLM_SAMPLE_SIZE}개 샘플)")
print(f"하이브리드:   {accuracy_score(y_true_llm, y_pred_hybrid):.4f} (단건 경보 {LLM_SAMPLE_SIZE}개 / FAISS {faiss_count}건 + Agent {agent_count}건)")

print(f"\n=== 운영 지표 요약 ===")
print(f"{'방식':<12} {'p50(ms)':>10} {'p95(ms)':>10} {'1000건 비용($)':>18}")
print(f"{'FAISS 단독':<12} {percentile(faiss_latencies, 50):>10.1f} {percentile(faiss_latencies, 95):>10.1f} {0.0:>18.4f}")
print(f"{'RAG (LLM)':<12} {percentile(rag_latencies, 50):>10.1f} {percentile(rag_latencies, 95):>10.1f} {rag_cost_1k:>18.4f}")
print(f"{'Agent':<12} {percentile(agent_latencies, 50):>10.1f} {percentile(agent_latencies, 95):>10.1f} {agent_cost_1k:>18.4f}")
print(f"{'하이브리드':<12} {percentile(hybrid_latencies, 50):>10.1f} {percentile(hybrid_latencies, 95):>10.1f} {hybrid_cost_1k:>18.4f}")
