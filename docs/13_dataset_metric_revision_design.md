# Phase 13 — 데이터셋·평가지표 재확정 (2026-08-13 교수 미팅 반영)

> 이 문서는 사람이 읽는 **설계·의사결정 기록**입니다. 실제 수치는 `experiments/results/*.json`,
> 그림은 `docs/figures/` 에서 확인하세요.
>
> **착수 배경 (2026-08-13 교수 미팅, 지시 2건)**
> 1. **평가지표** — "MCC 를 찾아보니 고등학생 논문이었다. 심사기관이 리젝할 수 있으니 최근 논문 기준의
>    더 신뢰도 있는 지표를 쓰자."
> 2. **데이터셋** — "CSIC 2010 은 2010년 데이터다. 심사기관이 오래된 데이터셋을 신뢰성 부족으로 볼 수
>    있고, 최신 데이터셋이면 결과도 더 좋을 수 있다. 최근 논문에서 자주 쓰는 것으로 교체하자."
>
> **진행 원칙**: 데이터셋과 지표를 **먼저 확정**한 뒤 아이디어 실험 → 측정. (Phase 4~12 의
> 실험 아이디어는 유지하되, 측정 기반을 갈아끼운다.)

---

## 0. 요약 (결론 먼저)

| 항목 | 기존 | Phase 13 확정안 |
|---|---|---|
| 주 데이터셋 | CSIC 2010 (2010) + Kaggle 페이로드 | **SR-BH 2020** (Comput.&Secur. 120:102788, 2022) |
| 외부 검증셋 | 없음 | **Data 2025 10(11):186** — 2025년 실운영 WAF 차단 트래픽 (시간축 일반화) |
| CSIC 2010 | 주 트랙 | **legacy 비교 기준으로 강등**(삭제 아님 — 선행연구 대조용) |
| 헤드라인 지표 | Macro-F1 + **MCC** + benign-evasion | **P/R/Macro-F1 + PR-AUC + TPR@FPR + pAUC + 경보부하 + ECE** |
| MCC | 헤드라인 | **완전 제거** (2026-08-13 확정, §1.1.1) |
| 단일 요약 기준 지표 | MCC | **Macro-F1** 로 이관 (CV 비교·캐스케이드 임계값 선택, §1.4) |
| 클래스 구성 | 4클래스 | **4클래스 유지** — CAPEC 라벨을 매핑해 접음 (2026-08-13 확정, §2.5) |
| 지표 근거 문헌 | MDPI Technologies 2026, arXiv 2512.19203 | **Arp et al., USENIX Security 2022 / CACM 67(11):104–112, 2024** (최상위 권위로 교체) ⚠️ **지표마다 출처가 다르다 — §1.6** |

⚠️ **비용**: 데이터셋 교체는 Phase 4~12 의 **측정값 전부를 무효화**한다(F1~F13, τ 선택값,
회피 기준선, CV 결과 — 현재 `experiments/results/` 에 141개 JSON). 아이디어·코드는 살고 **숫자만 재생산**된다.
자세한 재실행 범위는 §4.

---

## 1. 지표 재검토 (지시 ①)

### 1.1 먼저 짚어야 할 사실 — MCC 자체는 "고등학생 지표"가 아니다

교수님 지적의 **관찰**(어떤 고등학생 논문이 MCC 를 썼다)은 사실일 수 있으나, 그것은
**지표의 신뢰도에 대한 논거가 아니다.** 어떤 지표를 누가 썼는지는 그 지표의 통계적 성질을
바꾸지 않는다. 그리고 실제 근거는 정반대다:

> **Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger, Cavallaro, Rieck.**
> *Dos and Don'ts of Machine Learning in Computer Security.* **USENIX Security 2022**.
> 확장판: *Pitfalls in Machine Learning for Computer Security*,
> **Communications of the ACM 67(11):104–112, 2024**. doi:10.1145/3643456

이 논문은 보안 ML 방법론 비판의 **표준 인용문헌**(보안 4대 학회 + ACM 대표 저널)이며,
소수 클래스 유병률이 표집 편향으로 부풀려진 경우 **MCC 사용을 명시적으로 권고**한다.
우리 트랙이 정확히 그 조건이다(Normal 3.4k vs 공격 각 28~40k — 공격이 인위적으로 과대표집).

즉 심사에서 문제가 될 수 있는 것은 MCC 의 존재보다 (ㄱ) 근거로 MDPI 급 문헌만 달아둔 것과
(ㄴ) MCC 를 **단독 헤드라인**으로 세운 구성이다.

### 1.1.1 결정 — 완전 제거 (2026-08-13 확정)

위 근거를 제시한 뒤에도 **MCC 완전 제거**로 확정했다(교수 지시 이행). 기록으로 남기는 이유:

- 이 문서 §1.1 의 Arp et al. 근거는 **삭제하지 않고 보존**한다. 심사에서 "왜 권고 지표를 뺐나"라는
  역질문이 나올 경우의 대응 자료다.
- MCC 가 없어도 §1.2 의 신설 지표군(PR-AUC · TPR@FPR · pAUC · 경보부하)이 불균형 문제를
  **각각 더 구체적으로** 다룬다. 즉 방법론 공백은 신설 지표로 메워진다 — 단일 요약값 하나를
  잃는 것이 비용의 전부다.
- ⚠️ **MCC 제거는 보고 열 삭제로 끝나지 않는다.** MCC 가 코드의 **의사결정 기준**으로
  박혀 있다 → §1.4 에서 이관 작업을 정의한다.

### 1.2 실제로 한 조치 — 근거 교체 + 헤드라인 재편 + 운영지표 신설

**(가) 근거 문헌을 최상위 권위로 교체**

| | 기존 (docs/07 §1.3) | Phase 13 |
|---|---|---|
| 1차 근거 | MDPI *Technologies* 14(1):54, 2026 | **USENIX Security 2022 / CACM 2024** (Arp et al.) |
| 보조 근거 | arXiv:2512.19203 (미게재 프리프린트) | 위 두 편을 보조로 유지(도메인 판본) |

기존 두 편은 **삭제하지 않고 보조로 내린다.** 논지는 같고, 다만 심사 대응력이 다르다.

**(나) 헤드라인 지표 재편** — Arp et al. 권고를 그대로 따른다.

| 순서 | 지표 | 역할 | 상태 |
|---|---|---|---|
| 1 | Precision / Recall / **Macro-F1** | 선행연구 직접 대조(§1.3 참조 — 최근 논문이 F1 로 보고) | 기존 |
| 2 | **PR-AUC (AP, macro)** | 희귀 사건에서 ROC 보다 정보량 큼 | 기존 |
| 3 | **TPR@1%FPR · TPR@0.1%FPR** | 운영 지점 성능 — WAF 는 저 FPR 영역에서만 쓸 수 있다 | **신설** |
| 4 | **pAUC** (FPR≤0.01 구간 정규화 ROC-AUC) | Arp et al.: "ROC 는 감당 가능한 FPR 까지만 보라" | **신설** |
| 5 | **경보 부하** (배포 base rate 기준 100만 요청당 오탐 수) | Arp et al.: FPR 을 음성 클래스 base rate 와 함께 논하라 | **신설** |
| 6 | **ECE / reliability diagram** | 캐스케이드 τ 게이트가 곧 확신도 임계값 → 교정 품질이 RQ5 의 전제 | **신설** |
| 7 | Accuracy, plain ROC-AUC | 부록 | 강등 유지 |
| — | ~~MCC~~ | — | **제거**(§1.1.1) |

- 기존 `benign-evasion`(공격→Normal 오분류율)은 **유지**한다. 4·5번이 그것의 운영 해석판이다.
- 통계 검증: 5-fold CV + paired t-test + Holm 보정 **유지** + **부트스트랩 95% CI 신설**.
  ⚠️ 2026-08-23 확인 — **부트스트랩 CI 는 아직 구현이 없다**(§5 대기 목록).
- ⚠️ **3·4번은 test 셋 Normal 표본 수가 부족하면 성립하지 않는다** — 현 트랙에서 TPR@0.1%FPR 은
  측정 불가다. 성립 조건과 트랙별 실측 해상도는 **§1.7**, 외부셋에서의 가용 지표도 거기 있다.

⚠️ **6번(ECE)의 근거는 Arp et al. 이 아니다** — 2026-08-23 원문 대조로 확인했다. 출처와
필요성 논거를 모두 정정했으니 **§1.6 을 반드시 함께 읽을 것.** (이 표의 "Arp et al. 권고를
그대로 따른다"는 문장은 3·4·5번에만 해당한다.)

⚠️ **표의 각 행이 근거로 삼는 문헌이 다르다.** 지표별 1차 출처:

| 지표 | 1차 출처 |
|---|---|
| 3 TPR@FPR · 4 pAUC | Arp et al. **P7** (bounded AUC 권고) + pAUC 표준화는 **McClish 1989** |
| 5 경보 부하 | Arp et al. **P8** + 계보의 원점 **Axelsson, ACM CCS 1999** |
| 6 ECE | **Naeini et al. AAAI 2015** · **Guo et al. ICML 2017** (Arp et al. 아님) |

**(다) 구현 지점** — 지표 **추가**는 `src/eval/metrics.py` **한 곳**만 고치면 전 모델에 자동
반영된다(단일 진실 소스 규칙, CLAUDE.md). `compute_metrics` 에 `tpr_at_fpr` · `pauc` · `ece` 키를
추가하고 `attack_focused_metrics` 에 경보 부하를 얹는다. 베이스라인·CNN·ViT·캐스케이드가 같은
함수를 타므로 **재구현 불필요**. 지표 **제거**는 그렇지 않다 → §1.4.

### 1.3 최근 논문이 실제로 보고하는 지표 (확인 결과)

| 논문 | 게재 | 보고 지표 |
|---|---|---|
| WADBERT (Dual-channel BERT) | arXiv:2601.21893, 2026-01 | **F1** (CSIC2010 99.63%, SR-BH2020 99.50%) |
| E-WebGuard | **Comput. & Secur. 148, 2025** | Acc/P/R/F1 계열 |
| WAMM | arXiv:2512.23610, 2025-12 | Accuracy, 추론 속도(µs), TP 차단율 |

**→ 시사점 2개.**
1. 최근 논문의 공통 통화는 여전히 **F1** 이다 → 선행 대조를 위해 F1 을 1번에 둔다(위 표).
2. 이들은 **저 FPR 운영 지점·경보 부하·교정을 보고하지 않는다** → 우리가 3·4·5·6번을 넣으면
   심사 대응을 넘어 **방법론적 우위**가 된다. "지표를 바꿨다"가 아니라 "선행연구가 안 보는
   운영 지점을 본다"로 서술할 수 있다.

### 1.4 ⚠️ MCC 제거의 파급 — 기준 지표 이관 (놓치면 코드가 조용히 망가진다)

MCC 는 보고 열이 아니라 **자동 의사결정의 기준값**으로 코드에 박혀 있다. 단순 삭제하면
캐스케이드가 임계값을 못 고르고 CV 비교가 기준을 잃는다. 확인된 5개 지점:

| # | 위치 | MCC 의 역할 | 이관 조치 |
|---|---|---|---|
| 1 | `src/eval/metrics.py:88` | `mcc` 키 산출, `format_summary` 출력(302~304) | 키·출력·import(`matthews_corrcoef`) 삭제 |
| 2 | `src/eval/cv_compare.py:132` | `--metric` **기본값 `"mcc"`** = 5-fold paired t-test 의 기준 | 기본값 **`macro_f1`** 로, `choices` 에서 `mcc` 제거 |
| 3 | `src/models/cascade.py:78,232-249` | `EXIT_MCC_TOLERANCE=0.0011` + 조기종료 임계값 선택 규칙("val MCC 하락 ≤ tol 중 평균깊이 최소") | `EXIT_F1_TOLERANCE` 로 개명, 규칙을 **Macro-F1** 기준으로 재작성 |
| 4 | `src/models/cascade.py:664-690` | H7-1 판정("같은 test MCC 유지하에 1차 비용 ≥30% 감소") | 판정 기준을 Macro-F1 로 재서술 |
| 5 | `tests/test_cascade.py:77`, `tests/test_early_exit.py:179-186` | 스윕 행 픽스처가 `mcc` 키 사용 | 키 이름 갱신(현재 `mcc`=`macro_f1` 동일값으로 넣고 있어 수정은 기계적) |

**기준 지표는 Macro-F1 로 이관한다.** 근거: §1.3 대로 최근 논문의 공통 통화이고,
클래스별 성능을 균등 가중해 소수 클래스 실패가 드러나므로 MCC 의 역할을 가장 가깝게 대체한다.

### 1.5 구현 완료 (2026-08-13)

| 파일 | 변경 |
|---|---|
| `src/eval/metrics.py` | `mcc` 제거 / `attack_score` · `tpr_at_fpr` · `partial_roc_auc` · `attack_operating_points` · `calibration_metrics` · `alert_load` · `save_reliability_diagram` 신설. 기준값은 모듈 상수(`TARGET_FPRS` · `PAUC_MAX_FPR` · `DEPLOYMENT_ATTACK_PREVALENCES` · `ECE_N_BINS`) |
| `src/eval/cross_validate.py` | fold 요약에서 `mcc` 제거, 중첩 지표(`pauc` · `ece` · `benign_evasion_rate`)와 `tpr_at_fpr_*` 평탄화 추가. ⚠️ **일부 fold 에만 있는 지표는 요약에서 제외** — 부분 평균은 paired 비교를 오염시킨다 |
| `src/eval/cv_compare.py` | `--metric` 기본값 `macro_f1`, `choices` 에 신설 지표 추가. `LOWER_IS_BETTER`(ece·benign-evasion) 도입 → 순위 정렬 방향과 그림 x축 범위 보정 |
| `src/models/cascade.py` | `EXIT_F1_TOLERANCE` 로 개명, 조기종료 임계값 선택·H7-1 판정·콘솔 출력을 Macro-F1 기준으로 재작성. 교정 곡선 저장 추가 |
| `src/models/train.py`, `baseline_tfidf.py` | 교정 곡선(`cal_<tag>.png`) 저장 추가 |
| `tests/test_metrics.py` | **신규 24개** — 손계산 대조로 정의를 못박음. `mcc` 부활 방지 테스트 포함 |

**검증**: `pytest tests/ -q` → **166 passed**. `train.py --track csic_binary --model cnn --smoke`
로 실제 데이터 경로 통과 확인(콘솔에 TPR@FPR·pAUC·ECE 노출).

⚠️ **경보 부하가 왜 필요한지 실제로 드러난 예** (합성 입력 점검):
Accuracy 0.997 / Macro-F1 0.995 / PR-AUC 1.000 / ROC-AUC 1.000 인 모델도, FPR 0.03 에
배포 공격비율 0.1% 를 가정하면 **경보 정밀도가 0.032** 로 붕괴한다(100만 요청당 헛경보 29,970건
vs 진짜 1,000건). 기존 지표만 보고했다면 이 모델을 "완벽"으로 보고했을 것이다.

⚠️ **허용폭 0.0011 은 그대로 쓸 수 없다.** 이 값은 "단일 split 실행 간 MCC 흔들림 ±0.11pp"를
실측해 정한 것이다(docs/07 §2.5). 지표가 Macro-F1 로 바뀌고 데이터셋도 바뀌므로
**새 트랙에서 같은 방식으로 노이즈 폭을 재측정**한 뒤 값을 정한다. (동일 설정 반복 학습 → 표준편차)
그 전까지 조기종료 임계값 선택 결과는 신뢰하지 않는다.

### 1.6 ⚠️ 출처 검증 결과 (2026-08-23) — 원문 대조로 3건 정정

신설 지표 4종이 "실제로 그 논문에 있는 권고인지"를 원문으로 전수 대조했다. **3종은 확인,
1종(ECE)은 오귀속**이었다. 심사위원이 원문을 열어보면 드러나는 종류의 오류라 여기 남긴다.

**(가) 확인된 것 — Arp et al. 의 공신력과 인용 정확성**

서지 확정: **CACM 67(11):104–112, 2024, doi:10.1145/3643456**, ACM DL 의 **Research Highlights**
게재(= CACM 편집위 선별 지면). 원본은 USENIX Security 2022. 권위를 문제 삼기 어렵다.

| 우리 주장 | 원문 | 판정 |
|---|---|---|
| P7 이 저 FPR 구간 bounded AUC 를 권고 | *"we recommend considering the curves only up to **tractable false-positive rates** and to compute **bounded AUC values**"* | ✅ 정확 |
| P8 이 base rate 를 함께 논하라고 권고 | *"we advocate the use of precision and recall... these functions account for class imbalance"* | ✅ 정확 |
| **§1.1 의 "Arp et al. 는 MCC 를 권고한다"** | P8 권고 절: *"other measures like **Matthews Correlation Coefficient (MCC)** are more suitable"* | ✅ **정확 — §1.1 의 역질문 대응 자료는 그대로 유효** |

**(나) 정정 1 — ECE 는 Arp et al. 에 없다**

원문 전수 검색 결과 **calibration · calibration error · ECE · reliability diagram 언급 0건**이다.
그런데 `metrics.py` 헤더와 위 §1.2 는 신설 4종을 전부 "그 논문의 권고를 그대로 구현했다"로
묶어 서술하고 있었다. → 출처를 아래 3편으로 교체했다(코드 주석도 함께 수정 완료).

- **Naeini, Cooper, Hauskrecht. AAAI 2015** — ECE 정의의 원점
- **Guo, Pleiss, Sun, Weinberger. ICML 2017**, *On Calibration of Modern Neural Networks* —
  reliability diagram + ECE 를 현대 신경망 평가 표준으로 만든 논문. **"정확도는 올라갔지만
  현대 심층망은 더 이상 교정돼 있지 않다"**는 이 논문의 발견이 우리가 ECE 를 재야 하는 직접 논거다.
- **CalexNet (arXiv:2509.08318)** — 이미 `docs/thetics/` 에 보유. 조기종료 분기의 캘리브레이션을
  다루므로 **캐스케이드 도메인 판본**으로 맞는 인용이다(docs/12 §1 표에 등재됨).

**(다) 정정 2 — ECE 의 필요성 논거가 과대 주장이었다**

기존 문구: *"확신도가 교정되어 있지 않으면 게이트 자체가 근거를 잃는다."* → **정확하지 않다.**
τ 게이트에 필요한 것은 확신도의 **순위(ranking)** 이지 수치의 교정이 아니다. 확신도가 전부
0.99 쪽에 쏠려 있어도 val 에서 τ=0.995 로 잡히면 게이트는 정상 작동한다. 즉 심사에서
**"val 에서 τ 를 튜닝하는데 교정이 왜 필요한가"** 한 마디에 무너지는 논거다.

정정된 논거 3가지:
1. **τ 의 이식성** — 교정돼 있어야 val 에서 고른 τ 가 다른 데이터셋·배포 환경에서 같은 의미를
   갖는다. **Data 2025 외부셋 실험이 정확히 이 검증**이다.
2. **τ 의 해석 가능성** — "확신도 0.9 미만은 승급"이 실제로 "오답 확률 10%"를 뜻하는지.
3. **틀린 샘플에 과확신하는 실패의 진단** — 이건 순위 자체를 망가뜨리므로 게이트에 실제 타격이다.
   평균인 ECE 가 가리는 이 국소 실패를 **MCE** 가 드러낸다.

**(라) 정정 3 — 서지 보강 2편**

pAUC 와 경보 부하는 Arp et al. 이 "권고"했을 뿐 그 지표를 **만든** 논문이 아니다. 1차 출처를 병기한다.

- **McClish, D.K.** *Analyzing a portion of the ROC curve.* **Medical Decision Making 9(3):190–195, 1989.**
  — 우리가 쓰는 sklearn `max_fpr` 의 [0.5,1] 표준화가 이 논문의 보정이다(코드 주석엔 있었으나 문서엔 없었다).
- **Axelsson, S.** *The base-rate fallacy and its implications for the difficulty of intrusion
  detection.* **ACM CCS 1999.** doi:10.1145/319709.319710 — **"침입 탐지의 한계 요인은 탐지율이
  아니라 오탐률"**을 보인 이 계열의 원점. 넣으면 경보 부하 지표가 27년 축적된 표준 논거 위에 선다.

### 1.7 ⚠️ 저 FPR 지표의 성립 조건 — 현 트랙에서는 측정 불가

**FPR 의 해상도는 test 셋 Normal 표본 수가 결정한다.** Normal 이 N 개면 FPR 이 가질 수 있는 값은
1/N 의 배수뿐이다. 목표 FPR 이 1/N 보다 작으면 **그 지점은 존재하지 않고**, 코드는 "오탐 0건"
지점을 집어온다 → 사실상 "무오탐 탐지율"이 되고 표본 하나에 크게 흔들린다.

| 트랙 | test Normal | 해상도(1건당) | TPR@1%FPR | TPR@0.1%FPR |
|---|---|---|---|---|
| `payload_4class_csicnorm` (주) | **722** | 0.139pp | FP 7.2건 → 거침 | ❌ **정의 불가** |
| `csic_binary` | 726 | 0.138pp | 거침 | ❌ 불가 |
| `payload_4class` | 8,062 | 0.0124pp | ✅ | FP 8건 → 겨우 |
| **SR-BH 2020** (예정, test 15%) | **≈78,779** | 0.0013pp | ✅ | ✅ |

pAUC(FPR≤1%)도 같은 제약을 받는다 — 722개 트랙에서는 그 구간에 들어오는 음성이 약 7개뿐이라
**계단 몇 칸짜리 값**이 되어 비교 근거로 쓸 수 없다.

**→ 결론 2가지.**
1. 신설 지표 3종(TPR@FPR·pAUC·경보부하)은 **데이터셋 교체와 세트로만 성립**한다. 현 트랙에서
   미리 돌려 논문에 싣는 것은 위험하다. 이건 오히려 교수 지시 2건(지표·데이터셋)이 왜 한 묶음이어야
   하는지에 대한 논거가 된다.
2. 논문에 저 FPR 지표를 실을 때는 **Normal 표본 수를 반드시 함께 표기**한다.

**외부 검증셋(Data 2025)에서의 지표 가용성** — 악성만 있고 정상이 없다(§2.3).

| 지표 | 외부셋 |
|---|---|
| Recall(탐지율) · benign-evasion | ✅ |
| **ECE / MCE** | ✅ (확신도 vs 정답 여부만 필요) → **§1.6(다)-1 의 τ 이식성 검증 핵심** |
| TPR@FPR · pAUC · 경보부하 · FPR | ❌ 음성 표본이 없어 정의 불가 |

### 1.8 nan 누출 차단 (2026-08-23) — sklearn 판본 차이가 만든 조용한 오염

**증상**: `tests/test_metrics.py::test_pauc_returns_none_when_roc_is_undefined` 가 노트북에서
실패했다(`assert nan is None`). §1.5 에 "166 passed" 로 기록된 것은 데스크톱 기준이었다.

**원인**: sklearn 은 "계산 불가"를 **판본에 따라 두 방식**으로 알린다 — `ValueError` 를 던지거나,
`UndefinedMetricWarning` 과 함께 **`nan` 을 반환**하거나. 우리 코드는 `except ValueError` 로
앞의 것만 잡고 있었다. 같은 패턴이 `partial_roc_auc` · `_safe_roc_auc` · `_safe_pr_auc` **3곳**에 있었다.

⚠️ **이 버그가 위험한 이유는 죽지 않아서다.** nan 이 새면:
- `cross_validate` 의 fold 요약은 값이 수치형인지만 검사하므로 nan 이 통과 → **평균 전체가 nan**
  이 되거나, `cv_compare` 의 순위 정렬이 조용히 틀어진다.
- `json.dump` 는 nan 을 `NaN` 으로 쓰는데 이는 **표준 JSON 이 아니라** 다른 도구가 못 읽는다.
- 그리고 정상 표본이 없는 셋에서 터지므로, **Data 2025 외부 검증(§1.7)이 정확히 이 조건**이다.

**조치**: `_finite_or_none()` 헬퍼를 만들어 3곳에 적용하고, `partial_roc_auc` 에는
`tpr_at_fpr` 과 같은 단일 클래스 명시 가드를 앞에 뒀다. **예외 처리와 반환값 검사를 둘 다** 한다 —
sklearn 판본 고정으로는 노트북/데스크톱 양쪽 재현이 보장되지 않기 때문이다.

**검증**: 신규 테스트 2개(`_finite_or_none` 직접 검사 + `compute_metrics` 전 경로 nan 누출 검사)
추가 → `pytest tests/ -q` **168 passed**(기존 실패 1건 해소).

---

## 2. 데이터셋 재검토 (지시 ②)

### 2.1 CSIC 2010 에 대한 지적은 타당하다

문헌에서 확인된 CSIC 2010 의 실제 한계:
- **단일 도메인·단일 머신** 트래픽 → 개별 URL 단위 정확도만 평가 가능, 실배포 시나리오를 모사하지 못함
- 공격 유형 수가 매우 제한적, **세부 공격 유형 라벨이 없어** 선행연구가 대체로 **이진 분류**에 머묾
- 정상 트래픽이 **합성**(실사용자 행동이 아님)

즉 "2010년이라서"보다 **구성 자체의 한계**가 더 정확한 논거다. 논문에는 후자로 쓰는 게 강하다.

⚠️ 다만 **CSIC 2010 은 2025~2026 최신 논문에서도 여전히 보고된다**(E-WebGuard 2025,
WADBERT 2026 둘 다 CSIC 2010 + SR-BH 2020 병기). → **삭제하면 선행연구와 대조할 통화를 잃는다.**
그래서 교체가 아니라 **강등 + 병기**로 간다(§2.4).

### 2.2 채택 — SR-BH 2020 (주 데이터셋)

> Riera, Higuera, Higuera, Herraiz, Montalvo. *A new multi-label dataset for Web attacks CAPEC
> classification using machine learning techniques.* **Computers & Security 120 (2022) 102788**.
> doi:10.1016/j.cose.2022.102788 · 배포: Harvard Dataverse `doi:10.7910/DVN/OGOIXX`

| 항목 | 내용 |
|---|---|
| 수집 | 2020-07, 12일간. 인터넷에 **실제 노출된** WordPress 서버 |
| 라벨링 | ModSecurity 2.9.2 + **CRS 3.3.0** detection-only 로 기록 → **수동/반자동 검수로 교정** |
| 규모 | **907,814 요청** (정상 525,195 / 이상 382,619) |
| 구조 | 24 features + **13 라벨**(멀티라벨, CAPEC 분류) |
| 최근 사용 | **E-WebGuard**(Comput.&Secur. 148, 2025), **WADBERT**(2026-01), **WAMM**(2025-12) |

**우리 파이프라인 적합성 — 결정적으로, 원문 텍스트가 살아 있다.**
공개 노트북에서 확인한 실제 컬럼: `request_http_request`(URI), `request_body`,
`request_user_agent`, `request_cookie`, `request_referer`, `request_content_type`, 헤더 다수 —
**object(문자열) 타입**. 즉 바이트→48×48 이미지 변환 파이프라인이 **수정 없이 동작한다.**

⚠️ 원 논문의 *방법*은 필드별 ASCII 평균으로 24개 수치를 뽑는 것이지만, **배포된 데이터셋에는
원문 문자열이 포함**되어 있다. (이 구분을 놓치면 "인코딩된 수치뿐이라 이미지화 불가"로 오판한다.)

**13개 라벨** (CAPEC ID):
`000 Normal` · `272 Protocol Manipulation` · `242 Code Injection` · `88 OS Command Injection` ·
`126 Path Traversal` · `66 SQL Injection` · `16 Dictionary-based Password Attack` ·
`310 Scanning for Vulnerable Software` · `153 Input Data Manipulation` · `274 HTTP Verb Tampering` ·
`194 Fake the Source of Data` · `34 HTTP Response Splitting` · `33 HTTP Request Smuggling`

**부수 이득**: 현재 4클래스(Normal/SQLi/XSS/CmdI)의 빈약한 taxonomy 가 실공격 분류체계로 확장된다.
`payload_4class_csicnorm` 이 억지로 붙인 "합성 공격 + CSIC 정상" 조합도 **단일 출처 실트래픽**으로 대체된다.

### 2.3 채택 — Data 2025 (외부 시간축 검증셋)

> Lucz, G., Forstner, B. *A Thirty-Day Dataset of Malicious HTTP Requests Blocked by OWASP
> ModSecurity on a Production Web Server.* **Data 2025, 10(11), 186**. doi:10.3390/data10110186
> 배포: Zenodo `10.5281/zenodo.17178461` (`owasp.zip`, 29.5 MB, **CC-BY 4.0**)

- **2025년 실운영 서버**에서 OWASP CRS 가 실제 차단한 악성 요청 30일분
- 익명화하되 **method / URI / 헤더 / user-agent / 발동 rule ID 구조 보존** → 원문 바이트 사용 가능
- 포함 유형: SQLi, XSS, LFI, 스캐너 탐색, malformed·회피형 입력
- ⚠️ **악성만 있고 정상이 없다** → FPR 계산 불가. **탐지율(recall) 전용 외부 테스트셋**으로만 쓴다.

**왜 이게 지시 ②의 진짜 답인가**: 데이터셋을 2020년판으로 바꾸는 것만으로는 "그것도 6년 전"이라는
같은 지적을 다시 받는다. 대신 **2020년 데이터로 학습 → 2025년 실운영 공격에 대한 탐지율 측정**을
보고하면, "최신성"을 데이터 연도가 아니라 **측정된 일반화 결과**로 답하게 된다. 시간 경과에 따른
성능 감쇠(concept drift)는 그 자체로 기여 항목이 된다.

### 2.4 최종 데이터셋 구성

| 역할 | 데이터셋 | 연도 | 용도 |
|---|---|---|---|
| **주 트랙** | SR-BH 2020 | 2020 | RQ1·RQ2·RQ3·RQ5·RQ6 전부의 새 측정 기반 |
| **외부 검증** | Data 2025 (Zenodo) | 2025 | 시간축 일반화 탐지율(정상 없음 → recall 전용) |
| **legacy 대조** | CSIC 2010 | 2010 | 선행연구(E-WebGuard·WADBERT) 직접 대조용, 부록/비교표 |
| 페이로드 보강 | Kaggle SQLi-XSS-CmdI + PayloadsAllTheThings | — | RQ2 회피 변형 생성 재료로 유지 |
| 흐름 (RQ4b) | USTC-TFC2016 | 2016 | 이미 F1 0.9999 포화 → 부록 강등(기존 G6 결정 유지) |

### 2.5 클래스 구성 — 4클래스 유지 확정 (2026-08-13)

13개 CAPEC 라벨을 **기존 4클래스로 접는다.** 기존 실험 구조가 1:1 대응되어 "데이터셋만 교체"로
깔끔하게 서술되고, 재실행이 1회전으로 끝나는 것이 채택 이유다.

| SR-BH 라벨 (CAPEC) | 우리 클래스 |
|---|---|
| `000 Normal` | **Normal** |
| `66 SQL Injection` | **SQLi** |
| `242 Code Injection` | **XSS** |
| `88 OS Command Injection` | **CmdI** |
| 나머지 9라벨 (Path Traversal, Scanning, Verb Tampering, …) | 제외(또는 부록에서 별도 보고) |

⚠️ **매핑 규칙 3건을 코드로 못박고 문서화해야 한다** — 나중에 "왜 이 수치인가"를 재현할 근거다.
1. **`242 Code Injection` → XSS 는 완전한 동의어가 아니다.** CAPEC-242 는 코드 주입 일반이며 XSS 를
   포함하지만 더 넓다. → 클래스명을 `XSS` 로 쓸지 `CodeInj` 로 쓸지 결정하고, 논문에 매핑 표를 싣는다.
   (권고: **`CodeInj` 로 표기**하고 "XSS 를 포함하는 상위 범주"로 명시 — 이름을 XSS 로 두면 과대주장이 된다.)
2. **멀티라벨 → 멀티클래스 축약 규칙.** 한 요청에 여러 라벨이 붙은 경우의 처리를 명시할 것
   (권고: 대상 4클래스 중 2개 이상이 붙은 요청은 **모호 표본으로 제외**하고 제외 건수를 보고).
3. **제외된 9라벨 요청의 처리.** Normal 로 넣으면 안 된다(그건 공격이다) → **데이터셋에서 제외**하고
   제외 규모를 보고한다. ⚠️ 이걸 Normal 로 흘리면 §2.6 의 라벨 노이즈를 우리 손으로 재현하게 된다.

### 2.6 ⚠️ 알려진 리스크 — SR-BH 2020 의 라벨 노이즈

> Osama, Elebiary, Qassim, Amgad, Maghawry, Saafan, Ghalwash. *Enhanced Web Payload Classification
> Using WAMM: An AI-Based Framework for Dataset Refinement and Model Evaluation.*
> **arXiv:2512.23610**, 2025-12 (rev. 2026-01)

이 논문이 SR-BH 2020 을 감사한 결과 **정상으로 잘못 라벨된 악성 요청 48,522건**을 찾아냈다
(정상 525,195건의 **약 9.2%**). LLM 판정 후 웹 침투테스트 전문가가 300여 표본을 수동 검증
(LLM 판정과 100% 일치)했다고 보고한다. 정제 데이터셋의 공개 여부는 초록에서 확인되지 않았다.

**대응 방침**
1. 착수 시 **우리 자체 감사**를 먼저 돌린다(정규식 기반 SQLi/XSS/CmdI 시그니처로 정상 클래스 스캔).
   → 규모를 **우리 손으로 재현 확인**한 뒤 수치를 인용한다(다른 기기·다른 버전 가능성 대비).
2. **정제 전/후를 병기 보고**한다. 라벨 노이즈를 숨기지 않는 것 자체가 방법론 점수다.
3. ⚠️ 노이즈는 **정상 클래스에 악성이 섞인** 방향 → benign-evasion 과 FPR 을 **낙관적으로** 왜곡한다.
   우리 헤드라인이 정확히 그 지표들이므로 이 감사는 **선택이 아니라 필수**다.

---

## 3. 새 지표·데이터셋이 기존 RQ 에 미치는 영향

⚠️ **RQ 번호는 재배열하지 않는다**(CLAUDE.md — 파일명에 번호가 박혀 있음). 아래는 답의 갱신 여부다.

| RQ | 기존 답 | Phase 13 이후 |
|---|---|---|
| RQ1 (이미지 단독 vs 텍스트) | **"아니오"** — 이미지 열세 | ⚠️ **재측정 필요.** 답이 바뀔 수 있다. 바뀌어도 **원 문구와 함께 보고**(HARKing 금지 — CLAUDE.md) |
| RQ2 (회피 강건성) | 기준선 0.5981(RGB) | 새 트랙 기준선 재측정 |
| RQ3 (적대적 방어) | 방어본 τ 재선택 | 새 트랙에서 전 구성 재조립 |
| RQ4 (암호화 경계) | 정보이론적 경계 확인 | **영향 적음** — 엔트로피 스윕은 데이터셋 의존이 약함 |
| RQ5 (캐스케이드, 제안 모델) | 5.07배 속도 | τ 재선택 + **ECE 신설로 게이트 전제 보강** |
| RQ6 (표현 불일치) | 보조 지표로 하향 | 재측정 |

**추가 가능 항목(신규)**: 2025 외부셋 탐지율 = 시간축 일반화. 기존 RQ 에 없던 질문이므로
필요하면 **RQ7 로 뒤에 추가**한다(번호 재배열 금지 규칙 준수).

---

## 4. 재실행 범위 (비용 정직하게)

**살아남는 것** — 코드·아이디어·방법론 전부. 이미지 변환, 채널 인코더, 모델 6종, 캐스케이드,
조기종료, 회피 파이프라인, CV 프레임워크는 트랙만 갈아끼우면 그대로 동작한다.

**무효화되는 것** — `experiments/results/` 의 **141개 JSON 전부**와 그에서 파생된
`docs/figures/` 그림, F1~F13 실측 문구, τ 값, 회피 기준선.

**GPU 재실행 순서** (⚠️ GPU 1대 → **순차 실행 필수**, CLAUDE.md):
1. 데이터 준비: SR-BH 다운로드 → 라벨 감사(§2.6) → `preprocess.py` 새 트랙 → `build_image_dataset.py`
2. `metrics.py` 개편(§1.2 다) + 단위 테스트 → **여기서 `pytest tests/ -q` 통과 확인**
3. RQ1 본 측정(모델 6종 + TF-IDF 베이스라인) → 5-fold CV
4. 캐스케이드 τ 재선택 + ECE
5. RQ2/RQ3 회피·방어
6. Data 2025 외부 탐지율
7. legacy CSIC 2010 대조표(선행연구 비교용, 축소 구성)

**신규 코드 작업량** (트랙 추가 규칙 — CLAUDE.md "5곳"):
`preprocess.py`(TRACKS/REQUIRED_FILES) · `build_image_dataset.py` · `baseline_tfidf.py` ·
`diagnose_payload_bias.py` · `train.py` 의 `--track` choices 동시 갱신 + 다운로드 스크립트 신설
(`src/data/download_srbh.py`, `download_owasp2025.py`) + `.gitignore` 에 `data/raw/<신규>` 수동 추가.

---

## 5. 열린 결정 / 다음 단계

**확정됨 (2026-08-13)**
- [x] 클래스 taxonomy → **4클래스 매핑 유지**(§2.5)
- [x] MCC → **완전 제거**(§1.1.1), 기준 지표는 Macro-F1 로 이관(§1.4)
- [x] 주 데이터셋 → **SR-BH 2020** + 외부검증 Data 2025 + legacy CSIC 2010(§2.4)

**작업 대기 (순서대로)**
- [ ] SR-BH 2020 Dataverse 다운로드 → 파일 포맷·용량·라이선스 **직접 확인**
      (⚠️ 2026-08-13 이 기기에서 Dataverse API 조회가 네트워크 차단으로 실패 — 브라우저 또는 다른 기기에서 재확인)
- [ ] `src/data/download_srbh.py` 신설 + `.gitignore` 에 `data/raw/srbh2020/` 추가 후 `git check-ignore -v` 확인
- [ ] SR-BH 라벨 노이즈 자체 감사(§2.6) → 48,522건 규모 재현 확인
- [ ] CAPEC→4클래스 매핑 구현(§2.5 규칙 3건) + 제외 건수 보고
- [x] `metrics.py`: TPR@FPR · pAUC · 경보부하 · ECE 추가 / MCC 제거 → **완료(§1.5)**, 166 tests pass
- [x] 신설 지표 4종 **출처 원문 대조** → ECE 오귀속 정정 + 서지 2편 보강 + ECE 논거 하향 → **완료(§1.6)**
- [x] 저 FPR 지표의 표본 수 제약 규명 + 외부셋 가용성 정리 → **완료(§1.7)**
- [ ] ⚠️ **부트스트랩 95% CI 미구현** — §1.2 (나) 에 "신설"로 적혀 있으나 `src/` 전체에 `bootstrap`
      문자열이 0건이다(2026-08-23 확인). 문서만 앞서 나간 상태 → 구현하거나 문구를 내려야 한다
- [x] ⚠️ `partial_roc_auc` 의 `nan` 반환 버그 → **완료(§1.8)**, 168 tests pass
- [x] ⚠️ **Arp et al. 원문 PDF 확보** (2026-08-23) — USENIX Security 2022 판(19p, 오픈액세스)과
      CACM 2024 확장판(메타데이터 `Commun. ACM 2024.67:104-112` 로 서지 확인) 2편을
      `docs/thetics/` 에 받고 docs/12 §1.3 인벤토리에 등재. `git check-ignore` 로 미추적 확인 완료.
      ⚠️ 즉 **두 파일은 이 기기에만 있다** — 데스크톱에서도 작업하려면 다시 받아야 한다
      (URL: `usenix.org/system/files/sec22-arp.pdf`, `discovery.ucl.ac.uk/10212285/1/3643456.pdf`)
- [x] docs/07 §1.3·§3 에 Phase 13 대체 표시, 마스터 설계문서 §6.1 지표 목록 갱신 → **완료**
- [ ] Macro-F1 노이즈 폭 재측정 → `EXIT_F1_TOLERANCE` 값 확정(§1.4). ⚠️ **새 트랙 준비 후에 할 일**
- [ ] GPU 재실행 착수(§4 순서)
- [ ] 마스터 설계문서 §3 데이터셋 표를 §2.4 로 갱신(데이터셋 교체 실행 시점에)
- [ ] docs/12 §1 수집목록에 §6 인용문헌 6편 등재

---

## 6. 인용 문헌 (Phase 13 신규 — docs/12 §1 수집목록에도 등재 필요)

⚠️ `docs/thetics/` 는 .gitignore 대상 → **추적본은 docs/12 §1** (CLAUDE.md).

1. Arp et al. *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security 2022.
   확장판: *Pitfalls in Machine Learning for Computer Security.* **CACM 67(11):104–112, 2024.** doi:10.1145/3643456
2. Riera et al. *A new multi-label dataset for Web attacks CAPEC classification using machine learning
   techniques.* **Computers & Security 120:102788, 2022.** doi:10.1016/j.cose.2022.102788
3. Lucz, Forstner. *A Thirty-Day Dataset of Malicious HTTP Requests Blocked by OWASP ModSecurity on a
   Production Web Server.* **Data 10(11):186, 2025.** doi:10.3390/data10110186
4. Zhou, Yau, Gan, Liong. *E-WebGuard: Enhanced neural architectures for precision web attack detection.*
   **Computers & Security 148, 2025.** (CSIC 2010 + SR-BH 2020 병기 사례)
5. Osama et al. *Enhanced Web Payload Classification Using WAMM.* **arXiv:2512.23610**, 2025-12.
   (SR-BH 라벨 노이즈 48,522건 감사)
6. *WADBERT: Dual-channel Web Attack Detection Based on BERT Models.* **arXiv:2601.21893**, 2026-01.
   (최신 SOTA 대조 대상, F1 보고)

**§1.6 출처 검증으로 추가된 지표 1차 출처 4편** (2026-08-23)

7. Axelsson, S. *The base-rate fallacy and its implications for the difficulty of intrusion
   detection.* **ACM CCS 1999**, pp. 1–7. doi:10.1145/319709.319710
   (확장판: *ACM TISSEC 3(3):186–205, 2000*) — **경보 부하** 지표의 도메인 원점
8. McClish, D.K. *Analyzing a portion of the ROC curve.* **Medical Decision Making 9(3):190–195, 1989.**
   — **pAUC** 표준화(McClish 보정, sklearn `max_fpr` 이 쓰는 그것)의 1차 출처
9. Naeini, M.P., Cooper, G.F., Hauskrecht, M. *Obtaining Well Calibrated Probabilities Using
   Bayesian Binning.* **AAAI 2015.** — **ECE** 정의의 원점
10. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. *On Calibration of Modern Neural Networks.*
    **ICML 2017.** — reliability diagram + ECE 를 현대 신경망 평가 표준으로 만든 논문.
    "현대 심층망은 정확해졌지만 더 이상 교정돼 있지 않다"가 우리가 ECE 를 재는 직접 논거

> 이 4편은 **9·10번이 ECE 의 진짜 출처**라는 점이 핵심이다(Arp et al. 에는 교정 논의가 없다 — §1.6 나).
> 캐스케이드 도메인 판본인 CalexNet(arXiv:2509.08318)은 **이미 보유**하고 있으며 docs/12 §1 표에 등재돼 있다.
