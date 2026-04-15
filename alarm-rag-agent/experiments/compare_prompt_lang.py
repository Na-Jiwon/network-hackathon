"""
프롬프트 최적화: 한국어 vs 영문 프롬프트 tool-calling 안정성 비교

동일한 ReAct Agent 구조(STEP 1: search → STEP 2: refine → STEP 3: 분류)에서
프롬프트 언어만 다르게 하여 최종 분류 정확도(Accuracy)를 비교한다.

사용법: python compare_prompt_lang.py [샘플수]
기본: 100개 샘플
"""
import os
import sys
import time
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RETRY_WAIT = 10
MAX_RETRIES = 3

# === 데이터 로드 ===
df = pd.read_csv("data/Q2_train.csv")
tickets = df['ticketno'].unique()
train_tickets, test_tickets = train_test_split(tickets, test_size=0.2, random_state=42)
train_df = df[df['ticketno'].isin(train_tickets)]
test_df = df[df['ticketno'].isin(test_tickets)]

print(f"Train: {len(train_df)}개 / Test: {len(test_df)}개")
print(f"평가 샘플: {SAMPLE_SIZE}개\n")

# === 벡터스토어 로드 ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
if os.path.exists("faiss_eval_index"):
    vectorstore = FAISS.load_local(
        "faiss_eval_index", embeddings, allow_dangerous_deserialization=True
    )
else:
    train_unique = train_df[['alarmmsg_original', 'root_cause_type']].drop_duplicates()
    docs = list(train_unique['alarmmsg_original'])
    labels = list(train_unique['root_cause_type'])
    vectorstore = FAISS.from_texts(
        docs, embeddings, metadatas=[{"label": l} for l in labels]
    )
    vectorstore.save_local("faiss_eval_index")


def extract_label(text):
    for label in ["LinkCut", "PowerFail", "UnitFail"]:
        if label in text:
            return label
    return "Unknown"


def invoke_with_retry(func, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait = RETRY_WAIT * (attempt + 1)
                print(f"    rate limit, {wait}초 대기...")
                time.sleep(wait)
            else:
                raise e
    raise Exception(f"{max_retries}회 재시도 실패")


# === 도구 정의 (양쪽 동일) ===
def make_tools(_vectorstore):
    @tool
    def search_similar_alarms(alarm_message: str) -> str:
        """유사 경보 검색"""
        results = _vectorstore.similarity_search_with_score(alarm_message, k=3)
        filtered = [(doc, score) for doc, score in results if score < 0.8]
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
        filtered = [(doc, score) for doc, score in results if score < 1.2]
        if not filtered:
            return "재검색에서도 유사한 사례를 찾지 못했습니다. 알려진 패턴이 없는 경보입니다."
        result_text = f"[재검색 키워드: {keywords}]\n"
        for doc, score in filtered:
            label = doc.metadata.get("label", "Unknown")
            result_text += f"경보: {doc.page_content} | 장애유형: {label} (거리: {score:.4f})\n"
        return result_text

    return [search_similar_alarms, refine_search]


# === 영문 프롬프트 ===
PROMPT_EN = """You are a network fault classification expert.

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

# === 한국어 프롬프트 (동일한 의미) ===
PROMPT_KO = """당신은 네트워크 장애 분류 전문가입니다.

STEP 1: search_similar_alarms 도구를 경보 메시지로 1회 호출하세요.
STEP 2: 결과가 없으면 refine_search 도구를 1회 호출하세요.
STEP 3: 도구 결과를 기반으로 일반 텍스트 메시지로 답하세요.

중요한 규칙:
- 각 도구는 최대 1회만 호출하세요. 동일한 도구를 반복 호출하지 마세요.
- 사용 가능한 도구는 search_similar_alarms, refine_search 두 개뿐입니다.
- 다른 도구를 호출하지 마세요. 새로운 도구를 만들지 마세요.
- 최종 답변은 일반 텍스트여야 합니다 (도구 호출이 아님).
- 최종 답변은 정확히 한 단어여야 합니다: LinkCut, PowerFail, 또는 UnitFail.
- 설명하지 마세요. 다른 텍스트를 추가하지 마세요."""


def build_agent(prompt_text):
    llm = ChatGroq(model="llama-3.1-8b-instant")
    tools = make_tools(vectorstore)
    return create_react_agent(llm, tools, prompt=prompt_text)


def evaluate(agent, sample, label_name):
    """샘플에 대해 agent 실행하고 정확도 측정"""
    y_true, y_pred = [], []
    error_count = 0
    print(f"\n=== {label_name} 평가 시작 ({len(sample)}개) ===")
    for i, (_, row) in enumerate(sample.iterrows()):
        y_true.append(row['root_cause_type'])
        try:
            result = invoke_with_retry(lambda r=row: agent.invoke(
                {"messages": [("human", f"경보 메시지 '{r['alarmmsg_original']}'의 장애 유형을 분류해주세요.")]},
                {"recursion_limit": 6},
            ))
            pred = extract_label(result["messages"][-1].content)
        except Exception as e:
            error_count += 1
            print(f"  {i+1} 오류: {str(e)[:80]}")
            pred = "Unknown"
        y_pred.append(pred)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)} 완료 (현재 acc: {accuracy_score(y_true, y_pred):.4f})")
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{label_name}] Accuracy: {acc:.4f} / 에러: {error_count}건")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "label": label_name,
        "accuracy": acc,
        "errors": error_count,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# === 동일한 샘플로 양쪽 평가 (공정 비교) ===
sample = test_df.sample(min(SAMPLE_SIZE, len(test_df)), random_state=42)

agent_en = build_agent(PROMPT_EN)
result_en = evaluate(agent_en, sample, "영문 프롬프트")

agent_ko = build_agent(PROMPT_KO)
result_ko = evaluate(agent_ko, sample, "한국어 프롬프트")


# === 최종 비교 표 ===
print("\n\n" + "=" * 60)
print("프롬프트 언어별 tool-calling 안정성 비교 (최종 분류 정확도)")
print("=" * 60)
print(f"{'프롬프트':<15}{'Accuracy':<12}{'에러 건수':<12}{'샘플 수':<10}")
print("-" * 60)
print(f"{'영문 (EN)':<15}{result_en['accuracy']:<12.4f}{result_en['errors']:<12}{SAMPLE_SIZE:<10}")
print(f"{'한국어 (KO)':<14}{result_ko['accuracy']:<12.4f}{result_ko['errors']:<12}{SAMPLE_SIZE:<10}")
print("=" * 60)

# 결과 CSV 저장
out_df = pd.DataFrame({
    "alarm": list(sample['alarmmsg_original']),
    "true_label": result_en["y_true"],
    "pred_en": result_en["y_pred"],
    "pred_ko": result_ko["y_pred"],
})
out_df.to_csv("compare_prompt_lang_results.csv", index=False)
print(f"\n예측 결과 저장: compare_prompt_lang_results.csv")

summary_df = pd.DataFrame([
    {"prompt_lang": "영문 (EN)", "accuracy": result_en["accuracy"], "errors": result_en["errors"], "samples": SAMPLE_SIZE},
    {"prompt_lang": "한국어 (KO)", "accuracy": result_ko["accuracy"], "errors": result_ko["errors"], "samples": SAMPLE_SIZE},
])
summary_df.to_csv("compare_prompt_lang_summary.csv", index=False)
print(f"요약 저장: compare_prompt_lang_summary.csv")
