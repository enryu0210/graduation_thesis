# Phase 12 부속 — '비용 축' 선행연구 대조 (2026-08-06)

> 이 문서는 사람이 읽는 **설계·의사결정 기록**입니다.
> 배경: docs/11 §0 이 *"보고 단위는 캐스케이드"* 로 확정한 뒤, **"같은 계열에서 비용 개선을
> 주제로 한 선행연구가 있는가"** 를 조사한 결과.
> **⚠️ `docs/thetics/` 는 `.gitignore` 대상(27행)이라 PDF 는 추적되지 않는다.
> 이 문서가 수집 목록의 유일한 추적본이므로, 논문을 추가하면 여기에도 반드시 적는다.**
> **→ 전체 목록은 §1.3 이다**(§1·§1.1 은 Phase 12 의 비용 축 수집분만 다룬다).
> 2026-08-12 전수 대조에서 **6편이 어느 문서에도 없었다** — gitignore 라 기기를 옮기면
> 사라질 상태였다. §1.3 이 그 상환분이다.

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

### 1.1 추가 확보 (2026-08-09) — 진행보고서 인용 논문 일괄 수집

`docs/reports/progress_2026-08-08.pdf` 가 인용한 논문을 전수 대조해 **노트북 쪽 `docs/thetics/` 를
데스크톱과 동기화**하면서, §1.2 에 "유료"로 적혀 있던 ACM 2편을 함께 확보했다.

| 파일명 | 서지 | 왜 넣었나 |
|---|---|---|
| `acm_sac2025_earlyexit_ids.pdf` | Simioni, Viegas, Santin, Horhulhack. *An Early Exit Deep Neural Network for Fast Inference Intrusion Detection*. **ACM SAC '25**, pp.730–737. doi:10.1145/3672608.3707974 | **M4 의 직접 선행**(조기종료 IDS) |
| `acm_csur_earlyexit_survey.pdf` | Rahmath P. et al. *Early-Exit Deep Neural Network — A Comprehensive Survey*. **ACM CSUR**. doi:10.1145/3698767 | M4 관련연구 서술의 표준 인용 |

> ⚠️ **"ACM DL 유료"는 더 이상 참이 아니다.** 2026-08-09 확인 시 ACM DL 이 전면 Open Access 로
> 전환돼("ACM is now Open Access" 배너) 두 편 모두 로그인 없이 받혔다. 받는 법:
> `https://dl.acm.org/doi/pdf/<DOI>?download=true` 를 **브라우저로** 열면 즉시 내려받아진다
> (curl 은 봇 차단으로 403 — MDPI 도 동일하므로 브라우저 경로를 쓸 것).

### 1.2 확보 실패 — 수동 다운로드 필요

| 논문 | 사유 | 필요 이유 |
|---|---|---|
| *FastDet: Detecting Encrypted Malicious Traffic Faster via Early Exit*, Springer LNCS 2024, doi:10.1007/978-981-97-0834-5_18 | 유료(OpenAlex `is_oa=false`, 2026-08-09 확인) | M4 + RQ4(암호화) 교차 지점 |
| *Multi-Shield* = Robust image classification with multi-modal LLMs, **Pattern Recognition Letters 2025**, doi:10.1016/j.patrec.2025.04.022 | OA 로 표시되나 ScienceDirect 가 CAPTCHA 로 막음 | M1 의 최신 유사 사례(보조) |

→ 학교 도서관 프록시 또는 브라우저에서 직접 받을 것. **M4 를 포기하면 FastDet 은 불필요**(§4.3).

### 1.3 ⭐ `docs/thetics/` 전체 인벤토리 (20편 — 2026-08-12 전수 대조 18편 + 2026-08-23 Arp et al. 2편)

> ⚠️ **2026-08-24 갱신**: 이 표 이후 **13편이 더 들어왔다 → §1.4.** 현재 총 31편이다
> (20편 중 Arp et al. 2편은 노트북에만 있던 것이라 §1.4 에서 데스크톱에 다시 받았다 — 중복 아님).

⚠️ **§1·§1.1 은 Phase 12 의 '비용 축' 수집분만 적은 것이었다.** 그 앞 Phase 에서 모은 논문은
어느 문서에도 파일명이 없었고, `docs/thetics/` 가 gitignore 대상이라 **기기를 옮기면 조용히
사라질 상태**였다(실제로 6편이 그랬다 — 아래 🆕 표시). 여기서 전수 기록해 상환한다(G5 부분 상환).
**앞으로 논문을 추가하면 이 표에 적는다.**

| 파일명 | 서지 | 논문에서 쓰는 곳 |
|---|---|---|
| 🆕 `2016904.2016908.pdf` ⭐ | Nataraj, Karthikeyan, Jacob, Manjunath. *Malware Images: Visualization and Automatic Classification*. **VizSec 2011**. doi:10.1145/2016904.2016908 | **이미지화 계열의 원점.** 바이트-플롯 = 우리 R 채널(`raw_byte`)의 근거. 마스터 §10 |
| 🆕 `3510003.3510229.pdf` ⭐ | Wu, Zou, Dou, Yang, Xu et al. *VulCNN: An Image-inspired Scalable Vulnerability Detection System*. **ICSE 2022**. doi:10.1145/3510003.3510229 | 소스코드 이미지화 + **채널에 구문 정보**를 넣는 계열 → G 채널(`char_class`) 근거 |
| 🆕 `applsci-15-07163-v2.pdf` | Eroğlu Demirkan & Aydos. *Enhancing Malware Detection via RGB Assembly Visualization and Hybrid Deep Learning Models*. **Appl. Sci. 2025, 15, 7163**. doi:10.3390/app15137163 | "assembly-RGB 의 green 채널에 구문 정보" — `channel_encoders.py` 가 인용하는 그 논문 |
| 🆕 `EBSCO-FullText-2026. 07. 15..pdf` | Tadhani, Vekariya, Sorathiya, Alshathri, El-Shafai. *Securing web applications against XSS and SQLi attacks using a novel deep learning approach*. **Scientific Reports 14:1803 (2024)**. doi:10.1038/s41598-023-48845-4 | **텍스트 기반 웹공격 탐지 비교 대상**(CNN+LSTM 하이브리드). 마스터 §10 |
| 🆕 `technologies-14-00054-v2.pdf` ⭐ | Imani, Joudaki, Bagheri, Arabnia. *Why ROC-AUC Is Misleading for Highly Imbalanced Data: In-Depth Evaluation of MCC, F2-Score, H-Measure, and AUC-Based Metrics*. **Technologies 2026, 14(1), 54**. doi:10.3390/technologies14010054 | **지표 선택(MCC·PR-AUC)의 근거.** docs/07 §1.3 |
| 🆕 `2512.19203v2.pdf` | Thiyagarajan & Williams. *Evaluating MCC for Low-Frequency Cyberattack Detection in Imbalanced Intrusion Detection Data*. **arXiv:2512.19203v2**, 2026-01 | 위 지표 근거의 **침입탐지 도메인 판본**(CSE-CIC-IDS2017). docs/07 §1.3 보강 |
| 🆕 `arp_dos_donts_usenixsec22.pdf` ⭐⭐ | Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger, Cavallaro, Rieck. *Dos and Don'ts of Machine Learning in Computer Security*. **USENIX Security 2022** (19p) | **Phase 13 지표 개편의 1차 근거.** P7=bounded AUC(TPR@FPR·pAUC), P8=base rate fallacy(경보부하) + MCC 권고. docs/13 §1.6 |
| 🆕 `arp_pitfalls_cacm2024.pdf` ⭐⭐ | 위 확장판. *Pitfalls in Machine Learning for Computer Security*. **CACM 67(11):104–112, 2024**. doi:10.1145/3643456 (Research Highlights) | 위와 같은 논문의 저널 판 — **심사 대응 시 인용할 쪽**(ACM 대표 매체). docs/13 §1.6 |
| `시그니처 기반 필터링과 2D-CNN을…hybrid.pdf` | (국문) 시그니처 기반 필터링 + 2D-CNN 하이브리드 악성 트래픽 탐지 | 국내 선행. 캐스케이드 구조의 국문 대조 |
| `2312.13041v1.pdf` ⭐ · `mdpi_lightweight_cascade_zeroday.pdf` · `tiis_two_stage_waf_stacked_ensemble.pdf` | §1 표 참조 | 비용 축 캐스케이드 3편 |
| `deepsloth_2010.02432.pdf` ⭐ · `feature_squeezing_ndss2018.pdf` ⭐ | §1 표 참조 | F7·M1 의 직접 선행 |
| `calexnet_…2509.08318.pdf` · `p4sdn_…2509.12291.pdf` · `cascading_llm_failure_2605.17288.pdf` | §1 표 참조 | 조기종료·캐스케이드 붕괴 |
| `acm_sac2025_earlyexit_ids.pdf` · `acm_csur_earlyexit_survey.pdf` | §1.1 표 참조 | M4 의 선행(미채택이지만 관련연구에는 남김) |

> **`docs/reports/` 에 사본이 있는 5편**(`2312.13041v1` · `2512.19203v2` · `deepsloth_2010.02432` ·
> `feature_squeezing_ndss2018` · `s42400-023-00170-z`)은 **2026-08-12 부터 git 추적 대상**이다.
> 교수님 제출 묶음으로 복사해 둔 것이고, 커밋해 두면 다른 기기에서 다시 받을 필요가 없다.
> 나머지 13편은 여전히 `docs/thetics/`(gitignore)에만 있으므로 **이 표가 유일한 흔적**이다.

---

### 1.4 추가 확보 (2026-08-24) — Phase 13 지표 역검증에 쓴 논문 14편

docs/13 §1.9~§1.11 의 지표 역검증에서 실제로 대조한 논문이다. **지표 1차 출처**(위)와
**RQ 근접 이웃**(아래)으로 나눈다. ⚠️ 2026-08-23 에 노트북에서 받았다고 기록된 Arp et al. 2편은
**이 기기(데스크톱)에 없었다** — gitignore 라 따라오지 않는다는 것이 또 확인됐다. 이번에 다시 받았다.

| 파일명 | 서지 | 왜 넣었나 |
|---|---|---|
| `arp_dosdonts_usenixsec2022.pdf` ⭐ | Arp et al. *Dos and Don'ts of Machine Learning in Computer Security.* **USENIX Security 2022** (19p) | 현행 지표 체계의 1차 근거(P7 bounded AUC · P8 base rate) |
| `arp_pitfalls_cacm2024.pdf` ⭐ | 같은 저자 확장판. *Pitfalls in ML for Computer Security.* **CACM 67(11):104–112, 2024** | 위의 저널 판(심사 대응용 권위) |
| `axelsson_baserate_ccs1999.pdf` ⭐ | Axelsson. *The Base-Rate Fallacy and its Implications for the Difficulty of Intrusion Detection.* **ACM CCS 1999** (10p) | **경보 부하** 지표 계보의 원점 |
| `naeini_ece_aaai2015.pdf` | Naeini, Cooper, Hauskrecht. *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* **AAAI 2015** | **ECE** 정의의 원점 |
| `guo_calibration_icml2017_1706.04599.pdf` ⭐ | Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* **ICML 2017** | ECE·reliability diagram 을 표준으로 만든 논문 |
| `saito_prplot_plosone2015.pdf` ⭐ | Saito & Rehmsmeier. **PLOS ONE 10(3):e0118432, 2015** | **PR-AUC 의 유병률 의존성** 근거(docs/13 §1.10 발견 ①) |
| `calibration_metrics_review_2504.18278.pdf` | *A comprehensive review of classifier probability calibration metrics.* arXiv:2504.18278 (60p, 82개 지표) | ECE 의 한계와 대안(Brier) 검토용 |
| `advsqli_2401.02615.pdf` ⭐ | *AdvSQLi: Generating Adversarial SQL Injections against Real-world WAF-as-a-service.* arXiv:2401.02615 = **IEEE TIFS 게재본** | 🔴 **RQ2 최근접.** ASR 이 통화임을 원문 확인(34회) |
| `wafamole_2001.01952.pdf` | Demetrio et al. *WAF-A-MoLE: Evading WAFs through Adversarial ML.* arXiv:2001.01952 | RQ2 계보의 원점(변이 기반 우회) |
| `caliburn_2605.24696.pdf` ⭐ | *CALIBURN: Operationally Calibrated Streaming IDS with Regime-Dependent Conformal Risk Control.* arXiv:2605.24696 (58p) | 🔴 **경보 예산 α → FP 상한 임계값.** τ 선택의 경쟁 표준 · Brier 사용 |
| `wadbert_2601.21893.pdf` | *WADBERT: Dual-channel Web Attack Detection Based on BERT Models.* arXiv:2601.21893 | 같은 데이터셋(CSIC+SR-BH) 최신 SOTA. **F1 만 보고**함을 원문 확인 |
| `wamm_2512.23610.pdf` | *Enhanced Web Payload Classification Using WAMM.* arXiv:2512.23610 | SR-BH 라벨 노이즈 감사 근거 |
| `davis_goadrich_pr_roc_icml2006.pdf` ⭐ | Davis & Goadrich. *The Relationship Between Precision-Recall and ROC Curves.* **ICML 2006**, pp.233–240 | PR 곡선의 유병률 의존성 원점. ⚠️ 학교 미러는 전부 실패했고 **ACM DL 이 OA 라 `curl -A '<브라우저 UA>' 'dl.acm.org/doi/pdf/10.1145/1143844.1143874?download=true'` 로 받혔다** |
| `uncertainty_ensemble_deepkernel_2410.07725.pdf` | *Towards Trustworthy Web Attack Detection: An Uncertainty-Aware Ensemble Deep Kernel Learning Model.* arXiv:2410.07725 | 웹공격 탐지 + 불확실성. **Macro-F1 + weighted 병기**를 원문 확인 |

⭐ = 논문 본문에서 반드시 인용해야 하는 것.

**확보 실패 (재시도 대상)**

| 논문 | 사유 | 대안 |
|---|---|---|
| MDPI *Electronics* 14(21):4172 (멀티라벨 WAF) | Akamai 403 — curl·Chrome 헤드리스 모두 차단 | 브라우저 수동 다운로드 |
| RiskGate-IDS (**IJCIP 2026**, pii S187454822600048X) · E-WebGuard (**C&S 148, 2025**) | Elsevier 유료 | ⚠️ **원문 미확인 → 이 논문들을 근거로 단정 금지**(docs/13 §1.11 바) |
| *Towards Better-Calibrated ML Models for NIDS* (**IEEE WiMob 2025**, 11257482) | IEEE 유료 | 동상 |
| McClish. *Analyzing a portion of the ROC curve.* **Med Decis Making 9(3), 1989** | 유료 | pAUC 부록 강등 시 인용 부담도 사라짐 |

### 1.5 ⭐ 주제별 폴더 구조 (2026-08-24 재편) — 파일 경로의 기준

논문이 32편으로 늘어 한 폴더에서 찾기 어려워졌다. **기능·역할별 5개 폴더**로 나눴다.
⚠️ **§1~§1.4 표의 파일명 앞에는 아래 폴더가 붙는다.**

| 폴더 | 무엇을 모았나 | 편수 | 대표 논문 |
|---|---|---|---|
| `01_평가지표_방법론/` | **어떤 지표로 평가할 것인가**의 근거 — 보안 ML 평가 방법론 · 불균형 지표 · 교정(calibration) | 11 | Arp et al.(USENIX Sec'22 / CACM'24) · Axelsson(CCS'99) · Guo(ICML'17) · Saito(PLOS ONE'15) · Davis & Goadrich(ICML'06) · CALIBURN |
| `02_캐스케이드_조기종료_비용/` | **2단 구조·조기종료로 추론 비용을 줄이는 계열** — 제안 모델의 직접 선행 | 9 | Tasdemir(arXiv:2312.13041) · MDPI 경량 캐스케이드 · TIIS 2단 WAF · CalexNet · ACM SAC'25 · ACM CSUR 서베이 |
| `03_회피공격_방어/` | **탐지 우회 공격과 방어** (연구 질문 2·3) | 4 | AdvSQLi(IEEE TIFS) · WAF-A-MoLE · DeepSloth(ICLR'21) · Feature Squeezing(NDSS'18) |
| `04_이미지화_표현/` | **바이트를 이미지로 바꿔 분류하는 계열** — 제안 표현의 근거 | 4 | Nataraj(VizSec'11, 이미지화의 원점) · VulCNN(ICSE'22) · Appl.Sci. RGB assembly · fileless malware 이미지화 |
| `05_웹공격탐지_데이터셋/` | **웹공격 탐지 모델·데이터셋 최신 대조 대상** | 4 | WADBERT · WAMM · Uncertainty-Aware Ensemble · Sci.Rep. XSS/SQLi |

**분류가 애매해 판단이 필요했던 것 2건** (다음에 찾을 때 헤매지 않도록 남긴다)

- **CALIBURN** → `01`. 스트리밍 IDS 논문이라 `02` 도 가능하나, **우리가 쓰는 지점이 지표**다
  (Brier 사용례 + 경보 예산 α → FP 상한 임계값). 캐스케이드 τ 선택의 경쟁 표준으로도 인용한다.
- **CalexNet** → `02`. 교정 논문이지만 대상이 **조기종료 분기**라 캐스케이드 도메인 판본으로 쓴다.

> ⚠️ `docs/thetics/README.md` 에 같은 표를 두었으나 **그 파일도 gitignore 대상**이다(편의용 사본).
> 기기를 옮기면 사라지므로 **이 절이 유일한 추적본**이다.

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
