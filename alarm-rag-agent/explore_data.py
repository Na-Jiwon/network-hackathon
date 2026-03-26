import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

df = pd.read_csv("data/Q2_train.csv")

print("shape:", df.shape)
print("\n타겟 분포:")
print(df['root_cause_type'].value_counts())

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

test_alarms = ["ETH-ERR", "PSU-FAIL", "BATTERY_FAIL", "ETHER_LINK_DOWN"]
for alarm in test_alarms:
    print(f"\n=== 경보: {alarm} ===")
    results = vectorstore.similarity_search_with_score(alarm, k=10)
    for doc, score in results:
        print(f"거리: {score:.4f} (낮을수록 유사) | {doc.page_content[:60]}")

# threshold 설정 근거
# [train에 존재하는 경보] ETH-ERR, PSU-FAIL:
#   score 0.55~0.65 → 정확한 매칭
#   score 0.90 이상 → 유사하지만 장애유형 혼재 시작
#   → 1차 threshold 0.8: 신뢰할 수 있는 매칭만 필터링
#
# [train에 없는 B사 표현] BATTERY_FAIL, ETHER_LINK_DOWN:
#   가장 가까운 매칭도 score 1.22 이상
#   → 2차 threshold 1.2: 미학습 제조사 경보도 유사 사례 포함 가능
#
# 결론: 1차(0.8) → 2차(1.2) 2단계 fallback 구조로 설계