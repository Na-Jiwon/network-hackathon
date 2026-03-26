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

- [1. 데이터 탐색 및 핵심 문제 정의](#1-데이터-탐색-및-핵심-문제-정의)
- [2. 텍스트 정규화 및 시퀀스 설계](#2-텍스트-정규화-및-시퀀스-설계)
- [3. 모델링 및 성과](#3-모델링-및-성과)
- [4. 결론](#4-결론)

---

## 1. 데이터 탐색 및 핵심 문제 정의

전표(ticketno)별 경보 메시지를 **LinkCut / PowerFail / UnitFail** 3가지 장애 유형으로 분류하는 문제.
- 핵심 난이도: **Train에는 제조사 A만 존재, Test에는 A/B/C 존재 → 93%가 미학습 제조사**
- 같은 장애를 제조사마다 다르게 표현 → 정규화 전략 필수

| 제조사 A | 제조사 B | 제조사 C | 장애 유형 |
|---------|---------|---------|---------|
| PSU-FAIL | BATTERY_FAIL | Input Power Degrade Defect | PowerFail |
| ETH-ERR | ETHER_LINK_DOWN(LOS) | Loss Of Signal | LinkCut |
| OPT-LOS | LOSS_OF_SIGNAL | Loss Of Signal | LinkCut |

- Train 타겟 분포: LinkCut 50.18% / UnitFail 30.97% / PowerFail 18.85%
- 모델 입력: **alarmmsg_original** (경보 메시지 텍스트)
- 미사용 피처: unit, slot, port, sysname (장치 제조사마다 상이)

<!-- ![문제 정의](./images/파일명.png) -->

## 2. 텍스트 정규화 및 시퀀스 설계

- **표기 및 형식 통일**: 대문자 변환, 특수기호·구분자 통일, 불필요 구문 제거
- **도메인 용어 표준화**: 축약어를 풀어 제조사 간 공통 토큰 확보 → fastText 서브워드가 공통 토큰으로 유사성 학습 가능
- 총 **86개 매핑 규칙** 적용 (A사 30개, B사 32개, C사 24개)
- **ticketno 단위 시퀀스 피처 설계**: alarmlevel 오름차순 정렬 → 콤마로 경보 메시지 연결

<!-- ![텍스트 정규화](./images/파일명.png) -->

## 3. 모델링 및 성과

정규화 후에도 제조사별 변형이 남아 키워드·조합·문맥 세 관점에서 분류 → **3개 모델 앙상블**

- **fastText (Facebook AI)**: 서브워드 + 바이그램으로 키워드 패턴 학습, OOV·변형 표현 처리에 강점
- **FastText 임베딩 + LightGBM**: 임베딩 벡터 + 부스팅으로 비선형 조합 패턴 포착, class_weight 불균형 보강
- **Transformer (Keras)**: 소규모 도메인 어휘(125개)로 직접 구축, 시퀀스 순서·문맥 상호작용 학습

**Soft Voting 앙상블**: 3개 모델의 predict_proba 균등 평균(1/3) → 개별 모델 대비 예측 변동 완화

<!-- ![모델링 및 성과](./images/파일명.png) -->

## 4. 결론

- **최종 성과**: Accuracy 0.9448 (리더보드 3위 / 59팀)

<!-- 여기에 배운점이나 회고를 자유롭게 추가하세요 -->

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
<summary><b>데이터셋 상세</b></summary>

| 구분 | 크기 | 비고 |
|------|------|------|
| Train | 9,322 alarmno (1,114 ticketno) | 제조사 A만 존재 |
| Test | 37,671 alarmno (4,327 ticketno) | 제조사 A 7% / B 70% / C 23% |
| 단위 | alarmno (행 단위) | 하나의 ticketno에 여러 alarmno 포함 (평균 8.7개) |
| 타겟 | 장애 유형 3종 | LinkCut, PowerFail, UnitFail |

**결측치 현황**

| 구분 | slot | port |
|------|------|------|
| Train | 365개 (3.92%) | 597개 (6.40%) |
| Test | 2,781개 (7.38%) | 2,818개 (7.48%) |

</details>

<details>
<summary><b>주요 피처</b></summary>

| 피처 | 설명 | 사용 여부 |
|------|------|---------|
| alarmmsg_original | 경보 메시지 텍스트 | **모델 입력** |
| ticketno | 전표 번호 (경보 그룹 단위) | 집계 기준 |
| alarmtime | 경보 발생 시각 | 정렬 기준 |
| alarmlevel | 경보 심각도 | 정렬 기준 |
| unit, slot, port | 장치 위치 정보 | 미사용 (제조사마다 상이) |
| sysname | 장치 이름 | 미사용 (제조사마다 상이) |

</details>
