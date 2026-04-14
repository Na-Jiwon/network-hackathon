"""
BM25 + FAISS 하이브리드 검색 실험 (RRF fusion)

목적
----
경보 메시지는 `ETH-ERR`, `PSU-FAIL`, `ETHER_LINK_DOWN` 같은 영문 토큰 위주라
- dense (FAISS / mpnet): 의미적 유사도에 강함
- sparse (BM25): 정확 토큰 매칭에 강함

두 방식을 Reciprocal Rank Fusion(RRF)으로 결합하면 도메인 특성상 sparse
신호가 의외로 기여할 수 있는지 확인한다. 결과가 FAISS보다 낮더라도 **왜**
낮은지가 본 실험의 산출물이다.

RRF 공식:  score(d) = Σ_i  1 / (k + rank_i(d))
           (k=60 Cormack et al. 2009 원 논문 관례)

실행
----
    pip install rank_bm25
    python experiments/hybrid_bm25_faiss.py

같은 `random_state=42`, 같은 test 샘플, 같은 train/FAISS 인덱스를 사용해
`evaluate_faiss.py` 의 FAISS 단독 숫자와 직접 비교할 수 있게 맞췄다.
"""
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("rank_bm25 가 설치되지 않았습니다.  pip install rank_bm25")
    sys.exit(1)

load_dotenv()

# evaluate_faiss.py 와 완전히 같은 split을 사용해야 비교가 의미 있음
SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TOP_K = 3
RRF_K = 60  # 원 논문 관례값

# ---------- 데이터 로드 ----------
df = pd.read_csv("data/Q2_train.csv")
tickets = df["ticketno"].unique()
train_tickets, test_tickets = train_test_split(tickets, test_size=0.2, random_state=42)
train_df = df[df["ticketno"].isin(train_tickets)]
test_df = df[df["ticketno"].isin(test_tickets)]

train_unique = train_df[["alarmmsg_original", "root_cause_type"]].drop_duplicates().reset_index(drop=True)
corpus_texts = train_unique["alarmmsg_original"].tolist()
corpus_labels = train_unique["root_cause_type"].tolist()
print(f"Train unique: {len(corpus_texts)}개 / Test: {len(test_df)}개")

# ---------- 토크나이저 ----------
# 경보 코드는 `-`, `_`, `(`, `)` 등으로 분리된 영문 토큰 조합
# (예: ETH-ERR  →  ['eth', 'err'],  ETHER_LINK_DOWN → ['ether','link','down'])
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
def tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text)]

# ---------- BM25 인덱스 ----------
tokenized_corpus = [tokenize(t) for t in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

# ---------- FAISS 인덱스 (evaluate_faiss.py 와 같은 인덱스 재사용) ----------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
if os.path.exists("faiss_eval_index"):
    vectorstore = FAISS.load_local(
        "faiss_eval_index", embeddings, allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_texts(
        corpus_texts, embeddings,
        metadatas=[{"label": l} for l in corpus_labels],
    )
    vectorstore.save_local("faiss_eval_index")

# ---------- 평가 함수 ----------
def faiss_topk(query: str, k: int = TOP_K):
    """FAISS에서 top-k 문서 텍스트 리스트 반환 (rank 순)"""
    docs = vectorstore.similarity_search(query, k=k)
    return [(d.page_content, d.metadata.get("label")) for d in docs]


def bm25_topk(query: str, k: int = TOP_K):
    """BM25에서 top-k 문서 텍스트 리스트 반환 (rank 순)"""
    scores = bm25.get_scores(tokenize(query))
    # 상위 k index (내림차순)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(corpus_texts[i], corpus_labels[i]) for i in top_idx]


def rrf_fuse_docs(ranked_lists, k: int = TOP_K, rrf_k: int = RRF_K):
    """참고용: 문서 단위 RRF 융합 후 top-k 반환 (디버그 프린트에 사용)"""
    scores = defaultdict(float)
    label_map = {}
    for ranked in ranked_lists:
        for rank, (text, label) in enumerate(ranked):
            scores[text] += 1.0 / (rrf_k + rank + 1)
            label_map[text] = label
    fused = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [(t, label_map[t], s) for (t, s) in fused]


def rrf_label_vote(ranked_lists, rrf_k: int = RRF_K):
    """라벨 단위 RRF 투표: 각 리스트의 rank 가중치를 라벨별로 누적해 최대값 선택.

    작은 코퍼스에서 문서 단위 top-k 다수결은 tie-break 불안정 → 라벨별 가중합이
    안정적. RAG hybrid search 관행 중 하나 (label-level weighted fusion)."""
    label_scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (_text, label) in enumerate(ranked):
            if label is None:
                continue
            label_scores[label] += 1.0 / (rrf_k + rank + 1)
    if not label_scores:
        return "Unknown"
    # 동점이면 라벨 문자열 순(결정적) — LinkCut < PowerFail < UnitFail
    return max(sorted(label_scores.items()), key=lambda x: x[1])[0]


def majority_label(items):
    labels = [lbl for _, lbl in items if lbl is not None]
    if not labels:
        return "Unknown"
    return Counter(labels).most_common(1)[0][0]


# ---------- 실행 ----------
sample = test_df.sample(min(SAMPLE_SIZE, len(test_df)), random_state=42)
y_true = sample["root_cause_type"].tolist()
y_faiss, y_bm25, y_hybrid = [], [], []

for i, (_, row) in enumerate(sample.iterrows()):
    q = row["alarmmsg_original"]
    f = faiss_topk(q)
    b = bm25_topk(q)
    y_faiss.append(majority_label(f))
    y_bm25.append(majority_label(b))
    y_hybrid.append(rrf_label_vote([f, b]))

    # 첫 쿼리 한 건만 진단 출력: 겹침/라벨 분포를 눈으로 확인
    if i == 0:
        print("\n[debug] 첫 쿼리 융합 스냅샷")
        print(f"  query = {q!r}  gold = {row['root_cause_type']}")
        print(f"  FAISS top-3 = {f}")
        print(f"  BM25  top-3 = {b}")
        fused_docs = rrf_fuse_docs([f, b])
        print(f"  doc-level RRF top-3 = {fused_docs}")
        print(f"  label-vote 결과 = {y_hybrid[-1]}")


# ---------- 결과 ----------
def report(name, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name}  Accuracy: {acc:.4f} ===")
    print(classification_report(y_true, y_pred, zero_division=0))

report("FAISS 단독 (dense, mpnet)", y_faiss)
report("BM25 단독 (sparse, 토큰 매칭)", y_bm25)
report(f"Hybrid RRF (k={RRF_K})", y_hybrid)

print("\n=== 요약 ===")
print(f"FAISS  : {accuracy_score(y_true, y_faiss):.4f}")
print(f"BM25   : {accuracy_score(y_true, y_bm25):.4f}")
print(f"Hybrid : {accuracy_score(y_true, y_hybrid):.4f}")
print(
    "\n해석 가이드:\n"
    "- Hybrid > FAISS 이면 sparse 매칭이 미학습 제조사 경보 토큰에 기여.\n"
    "- Hybrid ≈ FAISS 이면 dense가 이미 sparse 신호를 포함해 fusion 이득 없음.\n"
    "- Hybrid < FAISS 이면 BM25 결과가 노이즈로 작용. 본 도메인은 라벨 분포가\n"
    "  편향돼 있어(LinkCut 우세) BM25 top-k가 다수결에서 FAISS 결정을 뒤집는 경우."
)
