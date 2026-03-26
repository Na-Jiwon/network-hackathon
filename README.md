# 통신망 안정화 AI 해커톤 프로젝트

통신망 안정성 확보를 위한 인공지능 해커톤 — **최우수상 수상** (전체 59팀 참가)

| 분야 | 과제 | 순위 |
|------|------|------|
| 분야1 | 무선 기지국 인구 밀집도 예측 | 59팀 중 **6위** |
| 분야2 | 유선 네트워크 경보 유형 분류 | 59팀 중 **3위** |

🔗 [대회 페이지](https://aifactory.space/task/2513/overview)

---

## 프로젝트 구성

### 📡 [wireless-traffic-prediction](./wireless-traffic-prediction)
무선 기지국 RU 통계 데이터를 기반으로 축제 지역 인구 수를 예측하는 회귀 모델

- **데이터**: 업/다운링크, BLER, RSSI, 단말 수 등 5분 단위 RU 통계
- **모델**: LightGBM, XGBoost
- **핵심**: 유의미한 피처 선택, 시계열 주기성 반영
- **평가**: MAE (평균 절대 오차)

### 🚨 [fault-alarm-classification](./fault-alarm-classification)
전표별 통신 경보 메시지를 링크/유니트/전원 장애로 분류하는 NLP 모델

- **데이터**: 경보 메시지, 발생 시각/위치, 장애 유형 레이블
- **모델**: FastText + LightGBM, Transformer 앙상블
- **핵심**: 제조사별 경보 표현 차이 처리 (수동 딕셔너리 86개 규칙), 경보 발생 순서 고려
- **평가**: Accuracy

### 🤖 [alarm-rag-agent](./alarm-rag-agent)
해커톤 분야2의 한계(수동 규칙 의존)를 **LLM + RAG**로 개선한 후속 프로젝트

- 기존 방식의 수동 딕셔너리 86개 규칙 → 규칙 없이 자동 분류
- **모델**: LLaMA 3.1 8B (Groq) + FAISS 벡터DB + LangGraph ReAct Agent
- **성능**: FAISS 단독 0.948 / RAG Agent 0.867 (단건 경보 기준)
- 새 제조사 추가 시 재학습 없이 자동 대응

---

## 기술 스택

| 영역 | 사용 기술 |
|------|----------|
| ML 모델 | LightGBM, XGBoost, FastText, Transformer |
| LLM / RAG | LLaMA 3.1 8B, LangChain, LangGraph, FAISS |
| 임베딩 | sentence-transformers/all-mpnet-base-v2 |
| UI | Streamlit |
| 언어 | Python |
