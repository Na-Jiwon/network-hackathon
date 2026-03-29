# 🔔 통신 경보 장애 분류 Agent

## 소개

ETRI·KT 통신망 안정화 AI 해커톤(전체 `158팀` 참가)에서 수행한 경보 분류 과제를 LLM + RAG로 개선한 프로젝트

해커톤 당시 제조사별 경보 표현 차이 문제를 **수동 딕셔너리 86개 규칙**으로 해결했으나 새 제조사가 추가될 때마다 규칙을 수작업으로 추가해야 하는 한계가 있었음. 이를 **규칙 없이 LLM + RAG로 자동 분류**하는 방식으로 개선.

→ [fault-alarm-classification](../fault-alarm-classification)(해커톤 분야2 `3위` / 59팀)의 수작업 한계를 AI 자동화로 대체한 확장 프로젝트

FAISS 기반 유사 경보 검색 → ReAct Agent의 동적 도구 호출 → LLM 분류 및 근거 생성으로 이어지는 RAG 파이프라인을 직접 설계·구현했다.

---

## 해커톤 방식과의 비교

### 핵심 문제

- Train: 제조사 A만 존재
- Test: 제조사 A, B, C 존재 → 미학습 제조사 `93%`
- 제조사마다 동일 장애를 다르게 표현

| 제조사 A | 제조사 B | 제조사 C |
|---------|---------|---------|
| ETH-ERR | ETHER_LINK_DOWN | Loss Of Signal |
| PSU-FAIL | BATTERY_FAIL | Input Power Degrade Defect |
| OPT-LOS | LOSS_OF_SIGNAL | Loss Of Signal |

### 방식 비교

| 항목 | 해커톤 방식 | 이 프로젝트 |
|------|------------|------------|
| 전처리 | 수동 딕셔너리 86개 규칙 작성 | 전처리 불필요 |
| 모델 | fastText + LightGBM + Transformer 앙상블 | LLM + RAG Agent |
| 새 제조사 대응 | 규칙 추가 수작업 필요 | 재학습 없이 자동 대응 |
| 분류 근거 설명 | 불가 | LLM이 자동 생성 |
| 정확도 | 0.9448 (분야2 리더보드 3위 / 59팀) | - |

---

## 데이터

- 출처: ETRI·KT 통신망 안정화 AI 해커톤
- Train: `9,322개` 경보 (제조사 A)
- Test: `37,671개` 경보 (제조사 A/B/C)
- 데이터는 보안상 제외 (.gitignore 처리)

---

## 시스템 구조

```
새 경보 입력
→ [Retrieval] search_similar_alarms: FAISS에서 유사 경보 검색 (threshold 0.8)
→ [Agent]    결과 부족 시 refine_search로 재검색 여부를 LLM이 스스로 판단 (threshold 1.2)
→ [Generation] LLM이 유사 사례 참고해 장애 유형 분류 + 이유 설명 생성
→ Streamlit UI로 결과 출력
```

| 단계 | 역할 |
|------|------|
| **Retrieval** | FAISS 벡터DB에서 입력 경보와 가장 유사한 학습 경보를 검색, threshold로 신뢰도 필터링 |
| **Agent** | ReAct(LangGraph) 구조로 검색 결과가 불충분하면 재검색 도구를 동적으로 추가 호출 — 단순 RAG와 달리 상황에 따라 검색 전략을 스스로 조정 |
| **Generation** | 유사 사례를 컨텍스트로 받은 LLM이 장애 유형 판단 + 근거 설명 생성 |

<img width="788" height="828" alt="rag" src="https://github.com/user-attachments/assets/af620ecc-46f0-4f0e-b3b1-dd3633b3a472" />

---

## Threshold 설정 근거

`explore_data.py`에서 유사도 분포를 실험해 설정.

| 경보 유형 | 거리 범위 | 해석 |
|---------|---------|------|
| Train에 존재하는 경보 (ETH-ERR 등) | 0.55~0.65 | 정확한 매칭 |
| 유사하지만 장애유형 혼재 시작 | 0.90 이상 | 신뢰도 낮음 |
| Train에 없는 B사 표현 (BATTERY_FAIL 등) | 1.22 이상 | 미학습 제조사 |

- **1차 threshold** `0.8`: 신뢰할 수 있는 매칭만 필터링
- **2차 threshold** `1.2`: 미학습 제조사 경보도 유사 사례 포함

---

<img width="431" height="585" alt="app" src="https://github.com/user-attachments/assets/e693ca86-e6e3-4e57-9d30-5a36b5e7928c" />

---

## 성능 비교

`evaluate_faiss.py`에서 수행.

| 방식 | Accuracy | 평가 조건 |
|------|----------|---------|
| FAISS 단독 | 0.9480 | 단건 경보 500개 |
| RAG (LLM) | 0.8667 | 단건 경보 100개|
| Agent | 0.8667 | 단건 경보 100개 |
| 해커톤 원본 | 0.9448 | ticketno 시퀀스 기반, 공식 리더보드 |

FAISS 단독과 해커톤 원본의 차이는 입력 단위가 다르기 때문. FAISS는 단건 경보 입력, 해커톤 원본은 ticketno 단위 시퀀스 입력.

---

## 기술 스택

- **LLM**: LLaMA 3.1 8B (Groq API)
- **벡터DB**: FAISS (로컬)
- **임베딩**: `sentence-transformers/all-mpnet-base-v2` (MTEB `57.8`점, sentence-transformers 계열 최상위 성능 기준 선정)
- **프레임워크**: LangChain, LangGraph (ReAct Agent)
- **UI**: Streamlit

---

## 분류 대상

- **LinkCut**: 네트워크 링크 장애
- **PowerFail**: 전원 공급 장애
- **UnitFail**: 장치 유닛 장애

---

## 프로젝트 구조

```
alarm-rag-agent/
├── agent.py          # 벡터DB 로드, ReAct Agent 구성 (도구 2개)
├── app.py            # Streamlit UI
├── evaluate_faiss.py # FAISS 단독 / RAG / Agent 성능 비교 평가
└── explore_data.py   # 유사도 분포 분석 및 threshold 설정 근거
```

---

## 결론

- **Rule-based → AI 자동화 전환**: 수작업 `86개` 규칙에서 재학습 없는 LLM+RAG 자동 분류로 전환했다. 새 제조사가 추가되어도 학습 데이터에 사례만 추가하면 대응 가능해 확장성이 크게 개선됐다.
- **동적 검색 전략이 핵심**: 단순 RAG가 아닌 ReAct Agent를 선택한 이유는 미학습 제조사 경보처럼 1차 검색으로 유사 사례를 찾지 못하는 케이스가 존재하기 때문이다. LLM이 검색 결과를 판단해 재검색 여부를 스스로 결정함으로써 경보 표현 다양성에 유연하게 대응했다.
- **한계와 방향**: RAG 방식(`0.8667`)은 해커톤 시퀀스 기반 방식(`0.9448`)보다 정확도가 낮다. 단건 경보 입력 vs ticketno 시퀀스 입력이라는 조건 차이가 있으며, ticketno 단위 시퀀스를 RAG에 통합하면 성능 격차를 줄일 수 있을 것으로 판단된다.

---

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```
