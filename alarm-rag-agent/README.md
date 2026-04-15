# 🔔 통신 경보 장애 분류 Agent

## 소개

ETRI·KT 통신망 안정화 AI 해커톤에서 수행한 경보 분류 과제를 LLM + RAG로 개선한 프로젝트

해커톤 당시 제조사별 경보 표현 차이 문제를 **수동 딕셔너리 86개 규칙**으로 해결했으나 새 제조사가 추가될 때마다 규칙을 수작업으로 추가해야 하는 한계가 있었음. 이를 **규칙 없이 LLM + RAG로 자동 분류**하는 방식으로 개선.

→ [fault-alarm-classification](../fault-alarm-classification)(해커톤 분야2 `3위` / 59팀)의 수작업 한계를 AI 자동화로 대체한 확장 프로젝트

FAISS 기반 유사 경보 검색 → ReAct Agent의 동적 도구 호출 → LLM 분류 및 근거 생성으로 이어지는 RAG 파이프라인을 직접 설계,구현

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

| 항목 | 해커톤 방식 | agent 프로젝트 |
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
- 실제 데이터는 보안상 제외. 대신 구조 확인용 더미 샘플(`data/sample_Q2_train.csv`, 20행)을 포함해 실 데이터 없이도 파이프라인을 끝까지 돌려볼 수 있다.

---

## 시스템 구조

LangGraph `StateGraph`로 3-branch 분류 파이프라인을 구성한다. 각 노드는 state에
중간 근거(`references`, `messages`)를 누적해 API와 Streamlit UI가 동일한 state를
바탕으로 분기별 근거를 그대로 렌더링할 수 있다.

```
            ┌──────────────────┐
  입력 ───▶ │ fast_path_faiss  │  FAISS k=3, threshold 0.8
            └────────┬─────────┘
                     │ route_by_confidence
          unanimous ─┤├─ unclear
                     ▼
   ┌──────────────────────┐   ┌──────────────────┐
   │ finalize_unanimous   │   │  agent_refine     │  ReAct + refine_search
   └──────────┬───────────┘   └─────────┬────────┘
              │                         │ route_after_agent
              │               done ─────┤├───── fallback
              │                         ▼                ▼
              │                         │        ┌───────────────┐
              │                         │        │ fallback_vote │  FAISS k=3, threshold 1.2
              │                         │        └───────┬───────┘
              ▼                         ▼                ▼
            END ◀──────────────────── END  ◀──────── END
```

| 경로 | 조건 | 비용 |
|------|------|------|
| **FAISS 만장일치** | k=3 검색 결과 라벨이 전부 같고 거리 < 0.8 | LLM 호출 0 |
| **Agent 판단** | 라벨 혼재 또는 threshold 초과 | ReAct LLM 1~2회 |
| **FAISS fallback** | Agent 예외/재귀 한계 초과 | LLM 호출 0, threshold 1.2 다수결 |

서빙은 FastAPI `/classify` 엔드포인트(`api.py`)로 노출되며 Streamlit UI와 동일한
`build_graph`를 공유한다. 매 요청마다 `{alarm, method, latency_ms, cached, tool_calls}`
JSON 로그 한 줄을 stdout에 남기고 동일 경보 반복 입력은 in-memory 캐시로 LLM 호출
없이 응답한다. `/health`·`/stats` 엔드포인트로 헬스체크와 운영 지표를 함께 제공.

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

### 운영 지표

`evaluate_faiss.py` 는 정확도와 함께 단건 분류 **p50/p95 latency**, **Groq 토큰 사용량**, **1000건당 예상 비용**을 같이 출력한다. 
---

## 실험

설계 선택의 근거

- [`experiments/README.md`](experiments/README.md) — 임베딩 모델(MiniLM vs mpnet), 프롬프트 한/영 구성, threshold 0.8/1.0/1.2 분포, 아키텍처 선택 이유를 표로 정리
- [`experiments/compare_prompt_lang.py`](experiments/compare_prompt_lang.py) — 동일 ReAct 구조에서 시스템 프롬프트 언어만 한/영으로 바꿔 tool-calling 안정성과 최종 정확도를 비교하는 재현 스크립트

---

## 기술 스택

- **LLM**: LLaMA 3.1 8B (Groq API)
- **벡터DB**: FAISS (로컬)
- **임베딩**: `sentence-transformers/all-mpnet-base-v2` (MTEB `57.8`점, sentence-transformers 계열 최상위 성능 기준 선정)
- **프레임워크**: LangChain, LangGraph (ReAct Agent)
- **UI**: Streamlit
- **API 서빙**: FastAPI + Uvicorn

---

## 분류 대상

- **LinkCut**: 네트워크 링크 장애
- **PowerFail**: 전원 공급 장애
- **UnitFail**: 장치 유닛 장애

---

## 프로젝트 구조

```
alarm-rag-agent/
├── agent.py                       # 벡터DB 로드 + ReAct Agent + LangGraph StateGraph
├── api.py                         # FastAPI `/classify` 엔드포인트 + 구조화 JSON 로깅 + 캐시
├── app.py                         # Streamlit UI (동일 build_graph 공유)
├── requirements.txt               # 의존성 명세
├── evaluate_faiss.py              # 3방식 정확도 + p50/p95 latency + Groq 비용 측정
├── explore_data.py                # 유사도 분포 분석 및 threshold 설정 근거
├── tests/
│   └── test_classify.py           # 3-branch 분기 pytest 커버리지
└── experiments/
    ├── README.md                  # 임베딩/프롬프트/threshold 비교 실험 표
    └── compare_prompt_lang.py     # 시스템 프롬프트 한/영 tool-calling 안정성 비교
```

---

## 결론


- **Rule-based → AI 자동화 전환**: 수작업 `86개` 규칙에서 재학습 없는 LLM+RAG 자동 분류로 전환했다. 새 제조사가 추가되어도 학습 데이터에 사례만 추가하면 대응 가능해 확장성이 크게 개선됐다.
- **동적 검색 전략이 핵심**: 단순 RAG가 아닌 ReAct Agent를 선택한 이유는 미학습 제조사 경보처럼 1차 검색으로 유사 사례를 찾지 못하는 케이스가 존재하기 때문이다. LLM이 검색 결과를 판단해 재검색 여부를 스스로 결정함으로써 경보 표현 다양성에 유연하게 대응했다.
- **한계와 방향**: RAG 방식(`0.8667`)은 해커톤 시퀀스 기반 방식(`0.9448`)보다 정확도가 낮다. 단건 경보 입력 vs ticketno 시퀀스 입력이라는 조건 차이가 있으며 ticketno 단위 시퀀스를 RAG에 통합하면 성능 격차를 줄일 수 있을 것으로 판단된다.

---

## 실행 방법

```bash
pip install -r requirements.txt

# 1) FastAPI 서버 (백엔드 API — 실서비스 연동 기준점)
uvicorn api:app --reload --port 8000
#    Swagger UI : http://localhost:8000/docs
#    Health     : GET  /health
#    분류       : POST /classify  {"alarm_message": "BATTERY_FAIL"}
#    지표       : GET  /stats

# 2) Streamlit UI (동일 build_graph 를 공유하는 데모)
streamlit run app.