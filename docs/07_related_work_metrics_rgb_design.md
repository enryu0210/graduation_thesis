# Phase 7 — 관련연구 리서치 · RGB 채널 강화 · 성과지표 개편 (2026-07-15 교수 미팅 반영)

> 이 문서는 사람이 읽는 **설계·의사결정 기록**입니다. 실제 수치는 `experiments/results/*.json`,
> 그림은 `docs/figures/` 에서 확인하세요.
> 착수 배경: 2026-07-15 교수 미팅. 세 가지 지시 — ① RGB의 G/B 채널 값 변경 실험,
> ② 더 정확한 측정을 위해 암호화 데이터셋 도입, ③(무엇보다) 관련 논문 리서치로 지표·방법론 개편.

---

## 0. 교수 지시 3항목 → 작업 매핑

| 지시 | 기존 설계와의 연결 | 이 Phase의 조치 |
|---|---|---|
| ① RGB G/B 채널 값 변경 | §4 Step3에 계획만 있고 **미구현**(1채널 grayscale였음) | RGB 파이프라인 구현 + **교체 가능한 채널 인코더**로 ablation 기반 마련 (§2) |
| ② 암호화 데이터셋 (정확한 측정) | RQ4a `entropy_sweep`(AES 이미 존재) | "암호화 = 측정 정밀도" 재해석 + RQ4b 흐름 데이터셋 점검 (§4) |
| ③ 관련연구·지표·방법론 | RQ1 지표(Acc/MacroF1/AUC) | 문헌 근거로 **MCC·PR-AUC·FPR** 추가, 방법론 정렬 (§3) |

---

## 1. 관련연구 리서치 요약 (③의 근거)

### 1.1 바이트/코드 → 이미지 채널 인코딩 계열
| 연구 라인 | 채널 구성 | 우리에게 주는 시사점 |
|---|---|---|
| Nataraj et al. (malware byte-plot) | 1채널 grayscale = raw byte | 우리 **R 채널의 표준 근거** |
| Assembly-RGB malware (MDPI Appl.Sci. 2025) | opcode/구문 정보를 채널로, "syntactic info in **green** channel" | **G=구문(문자클래스)** 배치의 직접 근거 |
| MDMC — Markov image (Comput.&Secur. 2020) | 바이트 전이확률 R/G/B 3채널 융합, Acc 99.2% | 다채널이 단채널 대비 특징 추출 이득 있음 |
| 엔트로피 이미지 계열 (entropy-image malware) | 32바이트 창 Shannon 엔트로피 이미지, grayscale/entropy dual-stack | **B=local entropy** 근거 + RQ4(암호화) 연결 |
| VulCNN (ICSE'22) | degree/katz/closeness **그래프 중심성 3채널** | "채널 = 서로 다른 관점"이라는 설계 철학의 정석 |

**결론**: 우리 기본 조합 **R=raw_byte / G=char_class / B=local_entropy** 는 세 계열(바이트플롯·구문채널·엔트로피이미지)
각각의 표준을 하나로 합친 것으로 문헌적 정당성이 있다. 교수 지시대로 G/B는 ablation 대상 → 인코더 교체 구조로 구현.

### 1.2 웹 공격(SQLi/XSS/CmdI) 탐지 지표 관행
- 다수 연구가 **Accuracy·Precision·Recall·F1·AUC** 를 보고(예: ASCII+CNN, CNN-LSTM 하이브리드 F1≈0.97~0.99).
- 그러나 이들 데이터셋은 대개 **균형에 가깝고**, 보고 정확도가 0.98+로 포화 → 우리 `payload_4class` 처럼
  Normal이 소수(csicnorm)거나 표면토큰 shortcut이 있는 세팅에서는 **Accuracy/F1이 실제 탐지력을 과대평가**한다.
  (이미 metrics.py 의 `attack_focused`(benign-evasion)로 부분 대응 중.)

### 1.3 불균형 보안 데이터 지표 모범사례 (핵심 발견)
- **"ROC-AUC는 고불균형에서 오해를 부른다"** (MDPI Technologies 2025): 클래스 불균형·비대칭 비용에서
  ROC-AUC가 분류기 간 실무적 차이를 가린다.
- **MCC (Matthews Correlation Coefficient)** 가 불균형에 강건 — "높은 MCC는 항상 높은 ROC-AUC를 함의하지만 역은 아님".
  MCC 최적화 모델이 Acc/AUC 최적화보다 정규화 MCC 우수(0.80 vs 0.78/0.73).
- **PR-AUC (Average Precision)** 가 다수 음성(TN) 상황에서 ROC보다 정보량 큼.
- 종합 권고 지표군: **Precision, Recall, F1, PR-AUC, MCC, G-Mean, FPR/FNR, Confusion Matrix.**

**→ 방법론 개편 결정**: RQ1 지표에 **MCC · PR-AUC(macro) · FPR** 를 추가하고, 논문 본문 헤드라인은
Accuracy 대신 **Macro-F1 + MCC + benign-evasion/FPR** 조합으로 보고. (Acc는 부차 지표로 강등.)

---

## 2. RGB 채널 강화 — 구현 (①)

### 2.1 구현물
- `src/imaging/channel_encoders.py` (신규): 채널 인코더 순수함수 + 이름→함수 레지스트리 `ENCODERS`.
- `src/imaging/payload_to_image.py`: `payload_to_rgb_image(text, side, encoders)` 추가(grayscale 함수는 보존).
- `src/imaging/build_image_dataset.py`: `--channels gray|rgb`, `--rgb-encoders R,G,B` 추가.
- `src/models/data_image.py`: RGB `.npz` 로드 + (N,H,W,3)→(N,3,H,W) 텐서화.
- `src/models/cnn.py`: 기존 `in_channels` 인자 활용(수정 불필요) — train.py가 채널 수 자동 추론해 전달.
- `src/models/train.py`: `--channels`, `--rgb-encoders` 추가, 산출물 tag에 `_rgb` 접미사.

### 2.2 채널 인코더 카탈로그 (교수 지시 = G/B 조합 실험)
| 이름 | 의미 | 기본 배치 |
|---|---|---|
| `raw_byte` | 원본 바이트(Nataraj) | **R** |
| `char_class` | ASCII 카테고리 밝기(특수문자=255 강조) | **G** |
| `local_entropy` | 지역 Shannon 엔트로피(창=8) | **B** |
| `bit_popcount` | 바이트 1-비트 수(밀도) | ablation 후보 |
| `structural_special` | 공격 특수문자 이진 마스크 | ablation 후보 |
| `byte_delta` | 직전 바이트 대비 변화량 | ablation 후보 |

조합 교체 예: `--channels rgb --rgb-encoders raw_byte,structural_special,byte_delta`
→ 파일명 자동 구분(`..._rgb-rb-ss-bd.npz`), 코드 수정 없이 ablation.

### 2.3 계획된 ablation (GPU 실행)
1. **gray(1ch) vs rgb-기본(rb/cc/le)** — RGB 강화가 Macro-F1·MCC를 올리나?
2. **G 채널 스왑**: char_class ↔ structural_special ↔ bit_popcount
3. **B 채널 스왑**: local_entropy ↔ byte_delta
각 셀은 `train.py --model cnn --channels rgb --rgb-encoders ...` 로 학습, metrics.py 통일 평가.

---

## 3. 성과지표 개편 (③) — 구현 계획
- `src/eval/metrics.py` `compute_metrics` 에 추가: `mcc`, `pr_auc_macro`(Average Precision, OvR macro),
  그리고 `attack_focused` 는 이미 FPR(normal_false_positive_rate) 제공 → 요약 출력에 노출.
- 모든 모델이 같은 함수로 계산되므로 **베이스라인/CNN/RGB 전부 자동 반영**(단일 진실 소스 유지).
- 논문 표: (Acc) · Macro-F1 · **MCC** · **PR-AUC** · benign-evasion · FPR 열로 재구성.

---

## 4. 암호화 데이터셋 (②) — 방향 정리
교수 "더 정확한 측정을 위해 암호화 데이터셋으로 바꾸자"의 두 해석과 우리 대응:

1. **RQ4a 재확인(내용 탐지의 경계)**: 이미 `entropy_sweep`(S0평문→S4 AES)로 "랜덤 암호화 시 재학습해도
   F1 0.94→0.40 붕괴"를 실측(06 문서 §5.5). RGB 파이프라인에도 같은 스윕을 얹어 "이미지 표현도 같은
   붕괴점"인지 확인 예정.
2. **RQ4b(신호 이동, 흐름 이미지)**: 공개 암호화 트래픽(USTC-TFC2016 1순위)을 flow→image로 변환해 동일 CNN으로
   악성/정상 흐름 탐지. 문헌(FlowPic, Wang 2D-CNN)이 방법론 근거. **데이터 접근성/라이선스 점검이 선행 과제.**

> 정직성 못박기(06 문서 계승): 완전 암호화 시 *내용* 탐지 불가는 결함이 아니라 **정보이론적 발견**.
> ②의 "정확한 측정"은 이 경계를 **RGB 3지표(엔트로피–분리도–F1) 곡선**으로 더 선명히 보이는 것으로 실현.

---

## 5. 열린 결정 / 다음 단계
- [ ] (GPU) gray vs rgb ablation 1라운드 → RGB 강화의 순효과 확인.
- [ ] metrics.py에 MCC/PR-AUC 추가(§3) 후 기존 산출물 재생성(재현 스크립트로).
- [ ] USTC-TFC2016 접근성·용량·라이선스 점검(②-2, RQ4b 선행).
- [ ] 관련연구 원문 서지정보 확정(§1 표의 연도/저자 재확인 후 인용).
- [ ] G/B 채널 최적 조합 확정 → 설계문서 §4 Step3를 "계획"에서 "확정"으로 갱신.
