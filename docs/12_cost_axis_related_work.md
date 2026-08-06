# Phase 12 부속 — '비용 축' 선행연구 대조 (2026-08-06)

> 이 문서는 사람이 읽는 **설계·의사결정 기록**입니다.
> 배경: docs/11 §0 이 *"보고 단위는 캐스케이드"* 로 확정한 뒤, **"같은 계열에서 비용 개선을
> 주제로 한 선행연구가 있는가"** 를 조사한 결과.
> **⚠️ `docs/thetics/` 는 `.gitignore` 대상(27행)이라 PDF 는 추적되지 않는다.
> 이 문서가 수집 목록의 유일한 추적본이므로, 논문을 추가하면 여기에도 반드시 적는다.**

---

## 0. 결론 — 세 줄

1. **비용 개선을 헤드라인으로 삼은 캐스케이드 웹공격 탐지 논문이 이미 존재한다**
   (Tasdemir et al. 2023, SQLi, **20배**). 우리 프레이밍과 거의 같고 배수는 저쪽이 크다.
2. 그러나 **라우팅 철학·표현 이질성·회피 축이 전부 다르다.** 차별화는 유지된다(§2.3).
3. 우리가 "발견"이라 부르던 것 두 개(**F7 비용 기반 공격**, **M1 표현 불일치**)에는
   각각 명확한 선행이 있다(DeepSloth ICLR'21, Feature Squeezing NDSS'18).
   **주장 문구를 낮춰야 한다**(§4). 낮춰도 기여는 남는다.

---

## 1. 이번에 `docs/thetics/` 에 추가한 논문 (7편)

| 파일명 | 서지 | 왜 넣었나 | 충돌/보완하는 우리 축 |
|---|---|---|---|
| `2312.13041v1.pdf` ⭐ | Tasdemir, Khan, Siddiqui, Sezer, Kurugollu, Yengec-Tasdemir, Bolat. *Advancing SQL Injection Detection for High-Speed Data Centers: A Novel Approach Using Cascaded NLP*. arXiv:2312.13041v1, 2023-12-20 (IEEE 투고). QUB + NVIDIA | **가장 가까운 선행.** 웹공격 페이로드 캐스케이드 + 비용 헤드라인 | 🔴 캐스케이드·비용 축 정면 충돌 |
| `mdpi_lightweight_cascade_zeroday.pdf` | Kutlimuratov et al. *A Lightweight Cascade-Based Farmework for Real-Time Zero-Day Attack Detection*. **Computers 2026, 15, 174**. doi:10.3390/computers15030174 (2026-03-08) | 2단 캐스케이드 + 마이크로초 지연·**470k req/s** 헤드라인, CPU-only | 🟠 비용 축 / 🟠 RQ4(암호화 호환) |
| `tiis_two_stage_waf_stacked_ensemble.pdf` | Sevri & Karacan. *Two Stage Deep Learning Based Stacked Ensemble Model for Web Application Security*. **KSII TIIS 16(2):632, 2022**. doi:10.3837/tiis.2022.02.014 | **2단 WAF**, 동기가 명시적으로 *"지연 없이 서비스"* | 🟠 비용 축 (단, 라우팅이 라벨 계층) |
| `deepsloth_2010.02432.pdf` ⭐ | Hong, Kaya, Modoranu, Dumitraş. *A Panda? No, It's a Sloth: Slowdown Attacks on Adaptive Multi-Exit NN Inference*. **ICLR 2021 (Spotlight)** | **F7(비용 기반 공격)의 직접 선행** | 🔴 F7 신규성 |
| `feature_squeezing_ndss2018.pdf` ⭐ | Xu, Evans, Qi. *Feature Squeezing: Detecting Adversarial Examples in DNNs*. **NDSS 2018**. doi:10.14722/ndss.2018.23198 | **M1(표현 불일치 → 이상 신호)의 직접 선행** | 🔴 M1 신규성 |
| `calexnet_earlyexit_calibration_2509.08318.pdf` | Aperstein & Apartsin. *CalexNet: Soft Cascade-Aligned Training and Calibration for Lightweight Early-Exit Branches*. arXiv:2509.08318 | 조기종료 분기의 **캘리브레이션** — F6(게이트 품질) 개선 수단 | 🟢 F6·M4 보완 |
| `p4sdn_ddos_earlyexit_2509.12291.pdf` | Karrakchou, Zniber, Sebbar, Ghogho. *Collaborative P4-SDN DDoS Detection and Mitigation with Early-Exit NNs*. arXiv:2509.12291 | 보안 도메인 조기종료 적용 사례 | 🟠 M4 |
| `cascading_llm_failure_2605.17288.pdf` | Sun, Chen, Li. *When Efficiency Backfires: Cascading LLMs Trigger Cascade Failure under Adversarial Attack*. arXiv:2605.17288, 2026-05 | **캐스케이드 구조가 적대적 공격 하에 붕괴**한다는 최신 사례 | 🟢 F7·F10 논의 보강 |

⭐ = 논문 본문에서 반드시 인용·대조해야 하는 것.

### 1.1 확보 실패 — 수동 다운로드 필요

| 논문 | 사유 | 필요 이유 |
|---|---|---|
| *An Early Exit DNN for Fast Inference Intrusion Detection*, **ACM SAC 2025**, doi:10.1145/3672608.3707974 | ACM DL 유료 | **M4 의 직접 선행.** M4 를 진행한다면 필수 |
| *FastDet: Detecting Encrypted Malicious Traffic Faster via Early Exit*, Springer LNCS 2024 | 유료 | M4 + RQ4(암호화) 교차 지점 |
| *Early-Exit DNN — A Comprehensive Survey*, ACM CSUR 2024, doi:10.1145/3698767 | 유료 | M4 관련연구 서술의 표준 인용 |
| *Multi-Shield* (Robust image classification with multi-modal LLMs), Pattern Recognition Letters | 유료 | M1 의 최신 유사 사례(보조) |

→ 학교 도서관 프록시로 받을 것. **M4 를 포기하면 앞의 3편은 불필요**(§4.3).

---

## 2. ⭐ Tasdemir et al. 2023 정독 결과 — 가장 중요

### 2.1 그쪽이 한 것

- **구조**: Viola-Jones 캐스케이드에서 착안. 1차 **Passive Aggressive Classifier**(TF-IDF 계열
  고전 ML) → 2차 **BERT-Electra base**.
- **라우팅**: 1차를 **양성 가중치 ×1000** 으로 학습해 재현율을 극단으로 올리고
  (**recall 0.9973, FPR 0.0157**), 결정함수 **임계값 −0.3** 으로 "조금이라도 의심스러운" 것을
  전부 양성 판정. **양성 판정된 것만** 2차로 넘겨 오탐을 걷어낸다.
- **에스컬레이션 비율**: OWASP A3 사전확률 p(C=1)≈3.3% 를 대입해 **p(D=1) ≈ 4.8%** 로 추정
  (우리 대표 운영점 7.99% 와 같은 자릿수).
- **성능**: accuracy **99.86%**, F1 **0.9981**, transformer 단독 대비 **20배** 빠름.
- **새 지표 `F1 Efficiency (FE)`**: F1 과 정규화 추론시간의 **가중평균**(가중치 α).
  α=1.00 / 0.98 두 시나리오로 35개 방법을 재랭킹 — α 를 조금만 낮춰도 transformer 계열이
  하위로 밀린다는 걸 보인다.
- **평가**: SQLi **이진**, SQL 문장 3만+ 건, A100 GPU(Colab), 10회 반복 평균.

### 2.2 그쪽이 **하지 않은** 것 (본문 전수 검색으로 확인)

- ❌ **회피·난독화·적대적 실험 전무.** `obfusc|evasi|adversar|bypass` 전수 검색 결과 실질 언급 0.
- ❌ **확신도(불확실성) 기반 라우팅 아님.** 클래스 기반(양성이면 승급)이다.
- ❌ **연속 운영점 곡선 없음.** FE 는 *방법들을 비교*하는 지표이지, 한 시스템의
  정확도–비용 곡선을 그리는 장치가 아니다. 고정 운영점 1개만 보고한다.
- ❌ 다중 클래스 아님(SQLi 이진), 유의성 검정 없음, 게이트 품질(oracle 격차) 분석 없음.

### 2.3 우리와의 대조표 — 이 표가 차별화의 근거다

| 축 | Tasdemir 2023 | 본 연구 |
|---|---|---|
| 1차/2차 **표현** | 텍스트 → 텍스트 (동종) | **이미지 → 텍스트 (이종)** ⭐ |
| 라우팅 기준 | **예측 클래스**(양성이면 승급) | **예측 불확실성**(max softmax < τ) |
| 운영점 | 고정 1개 | **τ 스윕 연속 곡선**, 양 끝점이 두 단독 모델로 수렴(단위 테스트로 고정) ⭐ |
| 과제 | SQLi 이진 | SQLi/XSS/CmdI/Normal **4-class** |
| 게이트 품질 | 미분석 | **oracle 대비 14.8배 과잉** 정량화 ⭐ |
| 회피 강건성 | 없음 | RQ2/RQ3 전 축 ⭐ |
| 통계 검정 | 없음 | 5-fold CV + Holm 보정 |
| 속도 배수 | 20× (vs BERT) | 5.07× (vs char-CNN) |

> ⚠️ **배수를 직접 비교하지 말 것.** 2차 기준 모델이 다르다(BERT vs char-CNN). BERT 가
> 훨씬 무거우므로 분모가 커서 배수가 크게 나온다. 논문에서 "우리가 5.07배라 저쪽 20배보다
> 못하다"는 식의 서술도, 그 반대도 성립하지 않는다. **기준 모델을 명시**해 병기한다.

### 2.4 라우팅 차이가 만드는 실질적 귀결 (F7 과 연결)

Tasdemir 방식은 **공격처럼 보이는 트래픽을 보내면 그대로 2차가 호출**된다 →
비용 기반 공격에 **구조적으로 더 취약**하다(공격자가 확신도를 조작할 필요조차 없다).
우리 확신도 게이트는 공격자가 **1차의 확신도를 떨어뜨려야** 하므로 한 단계 더 어렵다.
그럼에도 우리는 problem-space 변형만으로 **2.17배**를 달성했다(F7).

→ **논문 서술 방향**: *"클래스 기반 라우팅과 불확실성 기반 라우팅은 비용 공격면의 크기가
다르며, 후자가 더 좁지만 0 은 아니다"* 로 쓰면 대조와 발견이 한 문단에 들어간다.

---

## 3. 나머지 근접 선행 요점

### 3.1 Kutlimuratov et al. 2026 (Computers 15, 174)

- 흐름/통계 메타데이터만 사용 → **페이로드를 안 본다 → 암호화 트래픽 호환**을 장점으로 내세움.
- CSIC 2012(HTTP), UNSW-NB15, CSE-CIC-IDS2018. **공격 유형 분리 프로토콜**(테스트 공격 유형을
  학습에서 완전 배제)로 zero-day 평가.
- 0.002~0.006 ms, **470k req/s**, 메모리 6.2MB 미만, **GPU 없이**.

**우리에게 주는 함의 2가지**
1. 🔴 **RQ4 서술을 조정해야 한다.** 우리 RQ4 는 *"암호화되면 내용 기반 탐지가 붕괴하므로 신호가
   흐름으로 이동한다"* 인데, 이 논문은 **처음부터 흐름 메타데이터만** 쓴다. 즉 "흐름으로 가야
   한다"는 우리 결론의 **도착지에 이미 사람이 있다.** 우리 기여는 *도착지*가 아니라
   **"경계가 어디인지 정량 증명"**(F12: 고정키 XOR F1 0.9445 유지 → AES 0.3965 붕괴,
   즉 엔트로피가 아니라 키 랜덤화가 원인)임을 더 분명히 써야 한다.
2. 🟢 그들의 **공격 유형 분리 프로토콜**은 우리 S-A(변형 계열 held-out)와 같은 정신이다.
   docs/10 §6 의 S-0/S-A 설계를 **문헌 근거로 뒷받침**할 수 있다.

### 3.2 Sevri & Karacan 2022 (KSII TIIS 16(2))

- 1차 정상/비정상 → **2차에서만 공격 유형 분류**. 동기가 명시적으로 *"clients' requests can be
  served without time delay"*.
- **라우팅이 라벨 계층(binary→multiclass)** 이라 우리 τ 게이트와 메커니즘이 다르다.
  다만 **"비용 때문에 2단으로 나눈다"는 동기는 동일**하므로 관련연구에서 반드시 언급.
- 자체 데이터셋 **GAZIHTTP** 구축(97.43%) + ECML-PKDD(94.77%).

---

## 4. 이 조사가 요구하는 **주장 수정** — 반드시 반영

### 4.1 F7(비용 기반 공격) — "발견" → "도메인 이전"

DeepSloth(ICLR'21)가 이미 입력 적응형 네트워크에서 **지연 1.5~5배 증폭**, 적대적 학습으로
방어 제한적임을 보였다. 우리 2.17배는 그 범위 안이다.

- ❌ 쓰면 안 되는 말: "새로운 공격면을 발견했다"
- ✅ 쓸 말: *"DeepSloth 가 보인 slowdown 공격이 웹공격 탐지 캐스케이드에서도 성립함을 실측했다.
  단 DeepSloth 는 픽셀 단위 gradient 교란(feature-space, 화이트박스)인 반면, 본 연구는
  **실제로 동작하는 페이로드 변형(problem-space)** 만으로 같은 효과를 냈다 — 즉 모델 접근 없이
  가능하다."*

→ **이 재서술이 오히려 더 강하다.** 실전 WAF 우회 기법(주석 삽입·URL 인코딩)만으로 성립하므로.

### 4.2 M1(표현 불일치) — "새 발상" → "구조적 무상화"

Feature Squeezing(NDSS'18)이 **원본 vs 압축 입력의 예측 불일치**로 적대적 예제를 탐지한다.
메커니즘이 동일하다.

- ❌ "표현 불일치를 이상 신호로 쓰는 것은 새롭다"
- ✅ 세 가지로 차별화:
  1. 기존은 **같은 모델 + 다른 전처리**, 우리는 **다른 표현 + 다른 아키텍처**
  2. 기존은 탐지를 위해 **추가 forward pass 비용**을 지불, 우리는 캐스케이드가 이미 2차를
     돌리므로 **모드 A 에서 추가 비용 정확히 0** ⭐ ← M1 의 진짜 기여는 여기다
  3. 기존은 feature-space 적대적 예제, 우리는 **problem-space WAF 우회**

### 4.3 M4(다단 조기종료) — 우선순위 최하로 강등 권고

ACM SAC 2025 에 침입탐지용 early-exit 선행이 이미 있고, FastDet(암호화 트래픽 조기종료),
CSUR 서베이까지 있다. **구조는 완전히 기지 기법**이다.
M4 의 고유 논증은 *"게이트 최적화로 도달 불가능한 천장(7.29배)을 먼저 정량화하고 깬다"* 뿐인데,
이는 부수적이다.

→ **권고: M1 → M2 순으로 진행하고 M4 는 시간이 남을 때만.**
(✅ 2026-08-06 docs/11 §2 우선순위 표에 반영 완료. 교수 확인 후 확정.)

### 4.4 FE 지표 채택 검토 (신규 제안)

Tasdemir 의 **F1 Efficiency** 는 정확도–지연을 하나의 스칼라로 묶는다.
**경쟁 논문의 지표로 우리 캐스케이드를 평가해 보이면** 대조가 가장 깔끔해진다.
우리는 τ 스윕이 있으므로 **α 를 바꿔가며 최적 τ 가 어떻게 이동하는지**까지 그릴 수 있다 —
저쪽은 고정 운영점이라 못 하는 것이다.

→ 열린 결정: `src/eval/metrics.py` 에 FE 를 추가할지 (§5).

---

## 5. 열린 결정

- [ ] **FE(F1 Efficiency) 지표 채택 여부** — 채택 시 `metrics.py` 일원화 원칙에 따라 그곳에만 추가.
- [ ] **M4 진행 여부** — §4.3 근거로 강등 권고. 교수님 확인 필요.
- [ ] **RQ4 서술 재조정** — Kutlimuratov 2026 이 흐름 기반 탐지를 이미 하고 있으므로,
      우리 기여를 "경계의 정량 증명(F12)"으로 좁힐 것. docs/06 §5.5 와 함께 갱신.
- [ ] **유료 논문 4편 확보**(§1.1) — M4 포기 시 3편 불필요.
- [ ] 서지정보 최종 확인(docs/11 G5) — 위 표의 DOI·권호는 PDF 1쪽에서 직접 읽은 값이나,
      Tasdemir 는 **arXiv 프리프린트**이므로 IEEE 게재 여부를 논문 제출 전 재확인할 것.

---

## 6. 요약 — 이 문서가 확정한 것

1. 비용 축 선행은 **존재한다**(Tasdemir 2023). 숨기지 말고 **먼저 인용하고 대조**한다.
2. 차별화는 **표현 이질성 · 불확실성 라우팅 · τ 연속 스펙트럼 · 회피 축** 네 가지로 유지된다(§2.3).
3. F7·M1 의 주장 강도를 **낮춘다**(§4.1, §4.2). 낮춘 뒤에도 기여는 남으며, F7 은 오히려
   problem-space 라는 점에서 더 강해진다.
4. **M4 는 후순위로 강등** 권고(§4.3).
5. RQ4 는 "흐름으로 이동" 이 아니라 **"경계의 정량 증명"** 으로 초점을 좁힌다(§3.1).
