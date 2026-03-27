# 장애 경보 분류 (Fault Alarm Classification)

통신 장비 경보 메시지를 기반으로 장애 유형을 자동 분류하는 NLP 모델

**Accuracy 0.9448** — 59팀 중 3위

| 항목 | 내용 |
|------|------|
| 기간 | 2023.07 ~ 09 (3개월) |
| 인원 | 4인 팀 |
| 역할 | EDA 수행, 문제 정의 및 텍스트 정규화 방향 논의 |
| 대회 | ETRI·KT 통신망 안정화 AI 해커톤 |
| 평가 지표 | Accuracy (ticketno 단위) |

---

### 목차

- [1. 문제 정의](#1-문제-정의)
- [2. EDA 및 핵심 문제 발견](#2-eda-및-핵심-문제-발견)
- [3. 텍스트 정규화 및 시퀀스 설계](#3-텍스트-정규화-및-시퀀스-설계)
- [4. 모델링 및 성과](#4-모델링-및-성과)
- [5. 결론](#5-결론)
- [6. 배운점](#6-배운점)

---

# 1. 문제 정의

- 통신사업자의 네트워크망은 다양한 이기종 장치들로 구성되어 있고 장치 제조사마다 경보 메시지의 표현 방식이 다르다. 신규 장치를 수용할 때마다 정규화라는 적지 않은 수작업이 필요하며 장치의 실시간 상태를 신속히 파악하고 조치하는 것이 네트워크망 안정성 확보의 핵심이다.
- 전표(ticketno)별 경보 메시지를 **LinkCut / PowerFail / UnitFail** 3가지 장애 유형으로 분류한다. 경보 메시지를 구성하는 다양한 키워드의 의미와 조합을 이해하고 다수의 경보 간 발생 순서를 고려하는 것이 중요하다.
- 모델 입력: **alarmmsg_original** (경보 메시지 텍스트)
- 핵심 난이도: **Train에는 제조사 A만 존재, Test에는 A/B/C 존재 → 93%가 미학습 제조사**. 같은 장애를 제조사마다 다르게 표현하기 때문에 정규화 전략이 필수적이다.

---

# 2. EDA 및 핵심 문제 발견

| 구분 | 크기 | 비고 |
|------|------|------|
| Train | 9,322 alarmno (1,114 ticketno) | 제조사 A만 존재 |
| Test | 37,671 alarmno (4,327 ticketno) | 제조사 A 7% / B 70% / C 23% |
| 단위 | alarmno (행 단위) | 하나의 ticketno에 여러 alarmno 포함 (평균 8.7개) |
| 타겟 | 장애 유형 3종 | LinkCut, PowerFail, UnitFail |

| 구분 | slot 결측 | port 결측 |
|------|----------|----------|
| Train | 365개 (3.92%) | 597개 (6.40%) |
| Test | 2,781개 (7.38%) | 2,818개 (7.48%) |

- Train 타겟 분포: LinkCut 50.18% / UnitFail 30.97% / PowerFail 18.85%
- 미사용 피처: unit, slot, port, sysname (장치 제조사마다 상이)

| 제조사 A | 제조사 B | 제조사 C | 장애 유형 |
|---------|---------|---------|---------|
| PSU-FAIL | BATTERY_FAIL | Input Power Degrade Defect | PowerFail |
| ETH-ERR | ETHER_LINK_DOWN(LOS) | Loss Of Signal | LinkCut |
| OPT-LOS | LOSS_OF_SIGNAL | Loss Of Signal | LinkCut |

- 같은 장애임에도 제조사마다 표현이 완전히 다르다. → 텍스트 정규화로 제조사 간 공통 토큰을 확보해야 한다.

---

# 3. 텍스트 정규화 및 시퀀스 설계

### 3-1. 표기 및 형식 통일

- 대문자 변환, 특수기호·구분자 통일, 불필요 구문 제거

### 3-2. 도메인 용어 표준화

- 축약어를 풀어 제조사 간 공통 토큰 확보 → fastText 서브워드가 공통 토큰으로 유사성 학습 가능
- 총 **86개 매핑 규칙** 적용 (A사 30개, B사 32개, C사 24개)

### 3-3. ticketno 단위 시퀀스 피처 설계

- alarmlevel 오름차순 정렬 → 콤마로 경보 메시지 연결
- 하나의 ticketno에 포함된 여러 경보 메시지를 하나의 시퀀스로 결합해서 모델 입력으로 사용했다.

---

# 4. 모델링 및 성과

정규화 후에도 제조사별 변형이 남아 키워드·조합·문맥 세 관점에서 분류 → **3개 모델 앙상블**

- **fastText (Facebook AI)**: 서브워드 + 바이그램으로 키워드 패턴 학습, OOV·변형 표현 처리에 강점
- **FastText 임베딩 + LightGBM**: 임베딩 벡터 + 부스팅으로 비선형 조합 패턴 포착, class_weight 불균형 보강
- **Transformer (Keras)**: 소규모 도메인 어휘(125개)로 직접 구축, 시퀀스 순서·문맥 상호작용 학습
- 모든 모델에서 오버피팅이 발생해서 오버피팅을 방지하는 방향으로 파라미터를 튜닝했다.

**Soft Voting 앙상블**: 3개 모델의 predict_proba 균등 평균(1/3) → 모델별 특징이 다르기 때문에 상호보완을 위해 사용했다.

---

# 5. 결론

- **최종 성과**: Accuracy 0.9448 (리더보드 3위 / 59팀)
- 텍스트 정규화(86개 매핑 규칙)가 가장 큰 성능 기여 요인이었다. 미학습 제조사의 경보 메시지를 학습된 표현으로 변환하는 것이 모델 성능보다 중요했다.
- 모델 다양성(키워드·조합·문맥)을 확보한 뒤 Soft Voting으로 예측 변동을 완화했다.

---

# 6. 배운점

- **전처리가 모델 성능을 결정한다**: 정규화 없이 fastText를 돌리면 제조사 간 표현 차이로 성능이 크게 하락했다. 모델 튜닝보다 입력 데이터 품질이 성능에 미치는 영향이 더 컸다.
- **도메인 데이터 전처리 설계**: 제조사별 경보 표현을 수동 분석하여 86개 매핑 규칙을 구축했다. 미학습 제조사 B/C 대응 — 비정형 데이터를 모델이 학습 가능한 구조로 변환하는 과정이 핵심이었다.
- **완성 후 한계를 인식하는 것도 역량이다**: 해커톤 모델은 Accuracy 0.9448이지만, 새 제조사가 추가될 때마다 규칙 추가 + 재학습이 필요한 구조적 한계가 있다. 이를 인식하고 재학습 없이 미학습 제조사에 즉시 대응하는 RAG Agent로 개선하는 방향을 도출했다. → [alarm-rag-agent](../alarm-rag-agent)

---

### 실행 순서

| 순서 | 파일 | 내용 |
|------|------|------|
| 1 | EDA.ipynb | 데이터 탐색, 클래스 분포, 제조사별 표현 차이 분석 |
| 2 | fasttext_model.ipynb | FastText 기반 서브워드 학습 및 키워드 패턴 분류 |
| 3 | lightgbm_FastText_model.ipynb | FastText 임베딩 + LightGBM 최종 분류 |
| 4 | transformer_model.ipynb | Transformer 시퀀스 학습 |
| 5 | soft_voting_with_model_3.ipynb | 3개 모델 Soft Voting 앙상블 |

---

<details>
<summary><b>주요 피처</b></summary>

| 피처 | 설명 | 사용 여부 |
|------|------|---------|
| alarmmsg_original | 경보 메시지 텍스트 | **모델 입력** |
| ticketno | 전표 번호 (경보 그룹 단위) | 집계 기준 |
| alarmno | 경보 ID (최소 단위) | 행 식별 |
| alarmtime | 경보 발생 시각 | 정렬 기준 |
| alarmlevel | 경보 등급 분류 | 정렬 기준 |
| sva | 경보 심각도 | 미사용 |
| site | 경보 발생 지역 (익명화) | 미사용 |
| sysname | 장치 이름 (익명화) | 미사용 (제조사마다 상이) |
| unit, slot, port | 장치 위치 정보 | 미사용 (제조사마다 상이) |
| root_cause_domain | 장치 제조사명 (A/B/C) | 분석용 |
| root_cause_type | 장애 유형 (타겟) | **예측 대상** |

</details>

<details>
<summary><b>약어 정리</b></summary>

| 약어 | 의미 |
|------|------|
| AIS | Alarm Indication Signal |
| CPU | Central Processing Unit |
| E1 | E-carrier 1 |
| LLCF | Link Loss Carry Forward |
| LSP | Label Switched Path |
| NVRAM | Non-volatile Random Access Memory |
| OAM | Operation and Maintenance |
| OOV | Out of Vocabulary |
| PDH | Plesiochronous Digital Hierarchy |
| PDP | Power Distribution Panel |
| PSU | Power Supply Unit |
| PW | Pseudowire |
| UTP | Unshielded Twisted Pair |

</details>
