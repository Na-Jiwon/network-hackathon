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
from agent import load_agent, classify_alarm

load_dotenv()

# 샘플 수 설정 (커맨드라인 인자로 변경 가능: python evaluate_faiss.py 100)
LLM_SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 30
FAISS_SAMPLE_SIZE = 500
RETRY_WAIT = 10  # rate limit 시 대기 시간(초)
MAX_RETRIES = 3

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

for _, row in faiss_sample.iterrows():
    results = vectorstore.similarity_search(row['alarmmsg_original'], k=3)
    labels = [r.metadata['label'] for r in results]
    pred = Counter(labels).most_common(1)[0][0]
    y_true.append(row['root_cause_type'])
    y_pred_faiss.append(pred)

print(f"Accuracy: {accuracy_score(y_true, y_pred_faiss):.4f}")
print(classification_report(y_true, y_pred_faiss))


# === RAG (LLM) 평가 ===
print(f"\n=== RAG Agent 평가 ({LLM_SAMPLE_SIZE}개) ===")
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_sample = test_df.sample(min(LLM_SAMPLE_SIZE, len(test_df)), random_state=42)
y_true_llm = [row['root_cause_type'] for _, row in llm_sample.iterrows()]
y_pred_rag = []

for i, (_, row) in enumerate(llm_sample.iterrows()):
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
    except Exception as e:
        print(f"  {i+1} 오류: {e}")
        pred = "Unknown"
    y_pred_rag.append(pred)
    print(f"  {i+1}/{LLM_SAMPLE_SIZE} 완료")

print(f"Accuracy: {accuracy_score(y_true_llm, y_pred_rag):.4f}")
print(classification_report(y_true_llm, y_pred_rag, zero_division=0))


# === Agent 평가 ===
print(f"\n=== Agent 평가 (refine_search 포함, {LLM_SAMPLE_SIZE}개) ===")
agent = load_agent(vectorstore)
y_pred_agent = []

for i, (_, row) in enumerate(llm_sample.iterrows()):
    try:
        result = invoke_with_retry(lambda r=row: agent.invoke({
            "messages": [("human", f"경보 메시지 '{r['alarmmsg_original']}'의 장애 유형을 분류해주세요.")]
        }))
        pred = extract_label(result["messages"][-1].content)
    except Exception as e:
        print(f"  {i+1} 오류: {e}")
        pred = "Unknown"
    y_pred_agent.append(pred)
    print(f"  {i+1}/{LLM_SAMPLE_SIZE} 완료")

print(f"Accuracy: {accuracy_score(y_true_llm, y_pred_agent):.4f}")
print(classification_report(y_true_llm, y_pred_agent, zero_division=0))


# === 하이브리드 평가 ===
print(f"\n=== 하이브리드 평가 (FAISS 만장일치 + Agent fallback, {LLM_SAMPLE_SIZE}개) ===")
y_pred_hybrid = []
faiss_count = 0
agent_count = 0

for i, (_, row) in enumerate(llm_sample.iterrows()):
    try:
        result = invoke_with_retry(
            lambda r=row: classify_alarm(vectorstore, r['alarmmsg_original'], agent)
        )
        pred = extract_label(result["answer"])
        if result["method"] == "FAISS 만장일치":
            faiss_count += 1
        else:
            agent_count += 1
    except Exception as e:
        print(f"  {i+1} 오류: {e}")
        pred = "Unknown"
    y_pred_hybrid.append(pred)
    print(f"  {i+1}/{LLM_SAMPLE_SIZE} 완료")

print(f"Accuracy: {accuracy_score(y_true_llm, y_pred_hybrid):.4f}")
print(f"FAISS 만장일치: {faiss_count}건 / Agent 판단: {agent_count}건")
print(classification_report(y_true_llm, y_pred_hybrid, zero_division=0))


# === 최종 비교 ===
print(f"\n=== 최종 비교 ===")
print(f"FAISS 단독:  {accuracy_score(y_true, y_pred_faiss):.4f} (단건 경보 {len(faiss_sample)}개 샘플)")
print(f"RAG (LLM):   {accuracy_score(y_true_llm, y_pred_rag):.4f} (단건 경보 {LLM_SAMPLE_SIZE}개 샘플)")
print(f"Agent:       {accuracy_score(y_true_llm, y_pred_agent):.4f} (단건 경보 {LLM_SAMPLE_SIZE}개 샘플)")
print(f"하이브리드:   {accuracy_score(y_true_llm, y_pred_hybrid):.4f} (단건 경보 {LLM_SAMPLE_SIZE}개 / FAISS {faiss_count}건 + Agent {agent_count}건)")
