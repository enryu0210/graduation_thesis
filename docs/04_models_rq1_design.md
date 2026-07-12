# Phase 4 — 모델 학습·평가 설계/결정 문서 (RQ1: 탐지 성능)

> 이 문서는 사람이 읽는 **설계·의사결정 기록**입니다.
> 실제 실험 수치는 `experiments/results/*.json`(모델별 지표)과
> `docs/figures/models/`(Confusion Matrix)에서 확인하세요.

RQ1: *"페이로드를 바이트 이미지로 변환한 CNN이 기존 텍스트 기반 모델과 유사하거나
더 나은 탐지 성능을 보이는가?"* — 이를 검증하기 위해 **제안 CNN 1종 + 베이스라인 3종**을
**같은 데이터·같은 지표**로 학습·평가한다.

---

## 1. Phase 3이 넘긴 열린 결정 — 확정 내용

| 결정 | 확정 | 근거 |
|---|---|---|
| RQ1 주 트랙 | **`payload_4class` / raw / side=48** | 설계상 RQ1 주 데이터셋. CSIC·decoded·32/64 는 ablation 후순위 |
| 두 트랙 결합/전이 | **하지 않음(독립 평가)** | 라벨 공간·granularity 상이(Phase 3 리스크). 결합은 도메인 편차가 라벨과 얽힐 위험 |
| CSIC 이미지화 단위 | 현행 유지(URL 기준), Phase 4 주 실험에서 제외 | CSIC URL 중복 77.9% 이슈는 일반화 트랙에서 별도로 다룸 |
| 클래스 불균형 | **class weight (inverse frequency)** | oversampling 대비 데이터 팽창 없이 간단. CSIC(2-class) 에서 특히 중요 |

---

## 2. 비교 대상 (모두 동일 test 셋 / 동일 지표)

| 구분 | 모델 | 입력 | 파일 |
|---|---|---|---|
| **제안** | 얕은 CNN (Conv 32→64→128, BN, GAP, FC) | **48×48 바이트 이미지** | `src/models/cnn.py` |
| 베이스라인 ① | TF-IDF(char n-gram 2~4) + LogisticRegression / RandomForest | 페이로드 문자열 | `src/models/baseline_tfidf.py` |
| 베이스라인 ② | char-level CNN (1D 다중 커널) | 바이트 시퀀스 | `src/models/text_models.py` |
| 베이스라인 ③ | BiLSTM | 바이트 시퀀스 | `src/models/text_models.py` |

**비교의 공정성 — 핵심 설계**
모든 모델이 "같은 원재료(UTF-8 바이트)"를 쓰되 **표현 방식만** 다르게 했다.
- 제안 CNN: 바이트를 2D 이미지로 리셰이프
- char-CNN/BiLSTM: 같은 바이트를 1D 시퀀스로
- 시퀀스의 truncation/zero-padding 규칙과 길이 예산(기본 `max_len = 48×48 = 2304`)을
  이미지 트랙과 **일치**시켰다.
→ 성능 차이를 "이미지화 자체의 효과"로 해석할 수 있게 한다.

---

## 3. 학습 설정 (`src/models/train.py`)

- 손실: `CrossEntropyLoss(weight=class_weights)` — 불균형 보정
- 옵티마이저: Adam(lr=1e-3), batch=128
- **조기 종료**: val Macro-F1 기준 patience=5, best 가중치 복원 (과적합·시간 절약)
- 시드 고정(42)으로 실행 간 편차 축소
- **device 자동 감지**: `cuda` 있으면 GPU, 없으면 CPU
  → 이 노트북(CPU)에서도 돌지만, **실제 학습은 Colab/RTX 4080 등 GPU 권장**
- 산출물: `experiments/results/{tag}.json`(지표), `docs/figures/models/cm_{tag}.png`(혼동행렬),
  `experiments/checkpoints/{tag}.pt`(가중치). tag = `{track}_{model}_{text}`

### 평가 지표 (`src/eval/metrics.py`, 단일 진실 소스)
Accuracy · Macro-F1 · Weighted-F1 · 클래스별 P/R/F1 · Confusion Matrix ·
ROC-AUC(OvR macro) · 추론 처리량(samples/sec, WAF 배포 논의용).
**모든 모델이 이 모듈로 평가**되어 계산 방식 차이에 의한 왜곡을 배제한다.

---

## 4. 실행 순서 (재현)

```bash
# 0) 사전: Phase 3 산출물(data/processed, data/images)이 있어야 함

# 1) 베이스라인 ① — sklearn 만 필요, GPU 불필요 (이 노트북에서도 실행됨)
python src/models/baseline_tfidf.py --track payload_4class --clf logreg
python src/models/baseline_tfidf.py --track payload_4class --clf rf

# 2) 제안 CNN + 텍스트 베이스라인 — torch 필요 (GPU 권장)
python src/models/train.py --model cnn        # 제안 모델
python src/models/train.py --model charcnn    # 베이스라인 ②
python src/models/train.py --model bilstm     # 베이스라인 ③

# 스모크(코드 점검용, 작게): --smoke [--limit 500] [--max-len 256]
python src/models/train.py --model cnn --smoke
```

### Google Colab / GPU 데스크톱에서 돌리는 법
1. 저장소를 clone 하고 `pip install -r requirements.txt` (Colab 은 torch 기본 설치됨).
2. Phase 3 산출물을 만든다: `preprocess.py` → `build_image_dataset.py`
   (또는 `data/processed`·`data/images` 를 Drive 에서 복사).
3. 위 학습 명령을 그대로 실행 — `train.py` 가 `cuda` 를 자동 감지해 GPU 로 학습한다.
4. `--epochs`, `--batch-size` 는 GPU 메모리에 맞게 조정. BiLSTM 은 `--max-len` 이 크면
   느리므로 GPU 에서 돌리거나 값을 줄인다.

---

## 5. RQ1 실험 결과 (payload_4class / raw / side=48, clean test)

> GPU 전면 학습 완료. 5개 모델을 **동일 test 셋·동일 지표 모듈**(`src/eval/metrics.py`)로 평가한 결과.
> 지표 원본: `experiments/results/*.json`, 혼동행렬: `docs/figures/models/`.

| # | 모델 | 입력 | Accuracy | Macro-F1 | AUC(OvR) | 학습 비용(비고) |
|---|---|---|---|---|---|---|
| 1 | TF-IDF + LogReg | 문자열 | 0.9765 | 0.9775 | 0.9989 | features 85.6s + fit 30.6s (GPU 불필요) |
| 2 | TF-IDF + RandomForest | 문자열 | 0.9921 | 0.9924 | 0.9997 | features 87.4s + fit 102.9s |
| **3** | **제안 CNN (바이트 이미지)** | **48×48 이미지** | **0.9490** | **0.9496** | **0.9944** | **best epoch 14, 4.0s/epoch** |
| 4 | char-CNN | 바이트 시퀀스 | 0.9957 | 0.9958 | 0.9999 | best epoch 8, 29.6s/epoch |
| 5 | BiLSTM | 바이트 시퀀스 | 0.9960 | 0.9962 | 0.9998 | best epoch 25, 67.3s/epoch |

### 해석 — RQ1에 대한 정직한 답

RQ1은 *"이미지화 CNN이 텍스트 모델과 **유사하거나 더 나은** 성능을 보이는가?"* 였다.
**clean in-distribution 기준으로는 "약간 낮다"** 가 실측 결론이다.

- **제안 CNN이 5종 중 최하위**(Macro-F1 0.9496). 텍스트 베이스라인(char-CNN/BiLSTM/RF)보다
  약 4~5%p 낮다. "이미지화가 텍스트보다 우월하다"는 주장은 이 데이터에서 성립하지 않는다.
- 다만 **학습 효율은 압도적**: 제안 CNN은 4.0s/epoch 로 char-CNN(29.6s)·BiLSTM(67.3s) 대비
  7~17배 빠르고, best epoch(14)까지 총 학습시간도 가장 짧다. WAF 배포/재학습 관점의 실용 이점.
- **왜 텍스트 모델이 더 높은가는 7절과 직결**: 텍스트 모델은 페이로드의 표면 구두점 토큰
  (`(`,`)`,`|`,`alert` 등)을 그대로 특징으로 쓴다. 이 데이터셋은 그 토큰만으로 거의 분리되므로
  (7절 "과제가 쉬움") 텍스트 모델이 포화 성능에 도달한다. 이미지 CNN은 같은 정보를 2D 리셰이프
  후 학습해야 하므로 clean 지표에서 손해를 본다.
- ⚠️ **이 순위는 clean·in-distribution 한정**이다. 텍스트 모델의 높은 점수는 표면 토큰 shortcut에
  의존한 결과일 수 있고(7절), 이는 **RQ2(회피 공격)에서 무너질 가설**이다. "clean에서 이긴 모델이
  회피에서도 이기는가"가 RQ2·RQ3의 핵심 질문이 된다. → clean 순위를 최종 결론으로 쓰지 말 것.

### 검증·재현 메모
- 단위 테스트 `tests/test_models.py` 28건 통과(지표·클래스가중치·시퀀스 인코딩·모델 출력 shape).
- 재현: 4절 "실행 순서" 명령을 GPU 환경에서 그대로 실행.
- 남은 통계 검증(5-fold CV + paired t-test)은 6절 참조 — 위 표는 단일 split(seed=42) 결과다.

---

## 6. 리스크 / 열린 결정 (다음 단계로 넘김)

- [x] GPU 전면 학습 후 5개 모델 지표 표 확정 (5절) — **단일 split 완료**
- [ ] 통계 검증: 5-fold CV + paired t-test(설계 6.1) — 위 표는 아직 단일 split
- [ ] ablation: decoded vs raw, side 32/48/64, RGB 강화(설계 4장 Step 3)
- [ ] `csic_binary` 일반화 트랙 평가(이미지화 단위 URL vs 전체요청 재검토 포함)
- [ ] 최종 모델 확정 → Phase 5(RQ2 회피 공격)의 공격 대상 모델로 사용
- [ ] (운영) 브랜치 정리 — 현재 Phase 2~4 가 모두 `phase1-data-acquisition` 브랜치에
  커밋돼 있어 이름과 내용이 불일치. phase 별 브랜치 분리 또는 브랜치명 변경 검토 필요

---

## 7. ⚠️ 핵심 리스크 — payload_4class 고성능의 원인 진단 (실측)

TF-IDF+LogReg 가 clean test 에서 **AUC≈0.999 / Macro-F1≈0.977** 로 매우 높게 나온다.
이것이 실력인지 데이터 편향인지 판단하려고 실측했다.
재현: `python src/eval/diagnose_payload_bias.py` (2026-07 실행 기준).

**진단 결과**

1. **데이터 누수 아님** — train/test 를 정규화(소문자+공백축약)한 뒤 완전일치는 **0.3%(89/29,969)**.
   Phase 3 의 exact dedup 이 제대로 동작했고 near-duplicate 누수도 무시할 수준. → 수치는 "진짜"다.

2. **과제가 쉬움(표면 토큰 직교)** — logreg 상위 char n-gram 이 곧 "누가 봐도 아는 토큰":
   SQLi=`)`,`(`,`=` / XSS=`alert`,`%3`(=`<`),`id=` / CmdI=`find`,`|`,`$`,`-exec`.
   char n-gram 이 이 구두점을 그대로 잡아 선형 모델도 거의 완벽 분리(이 데이터셋류의 알려진 포화 현상).

3. **가장 중요한 편향 — Normal 클래스가 '영어 산문'** — Normal 예시가 영화 리뷰류 자연어
   (예: *"I'm a huge fan of war movies..."*)인 반면 공격 3종은 기호 범벅. 즉 모델이
   *'공격이냐 정상이냐'* 가 아니라 *'문장이냐 기호냐'* 를 배우는 **shortcut** 위험.

**함의 (논문 4·5장·한계에 반드시 서술)**
- 이 AUC 는 **낙관적으로 부풀려진 값**이며, 실 HTTP 트래픽(정상 요청도 구조적) 에는 그대로 통하지 않는다.
  → Phase 3 에서 경고한 **도메인 편차** 가 여기서 실측됨.
- 오히려 이것이 **RQ2(회피 공격) 의 존재 이유**: clean in-distribution 지표는 다 높지만 회피/실트래픽에서
  무너지는 것을 보이는 게 기여. 즉 높은 clean AUC 는 예상된 결과이자 연구 동기다.
- AUC(0.999) > accuracy(0.977) 는 AUC 가 관대한 랭킹 지표라 자연스러운 현상.

**후속 조치 후보**
- [x] Normal 클래스를 **CSIC 실트래픽 정상 요청**으로 교체 → 아래 7.1에서 실측(결과: 편향 안 줄어듦)
- [ ] near-duplicate 기준(정규화/유사도) dedup 을 exact dedup 에 추가할지 검토

---

## 7.1 Normal 교체 실험 — "장르 편차가 원인인가?" 실측 (2026-07)

7절의 3번(Normal='영어 산문' shortcut)이 고성능의 **주 원인인지** 검증하려고,
공격 3종은 그대로 두고 **Normal 만 CSIC 2010 실트래픽의 전체 쿼리스트링으로 교체**한
새 트랙 `payload_4class_csicnorm` 을 만들어 같은 진단을 돌렸다.
(구현: `src/data/preprocess.py::load_payload_4class_csicnorm`, 재현: `--track payload_4class_csicnorm`)

**설계상 유의점 — 왜 '전체 쿼리스트링'인가**
- 단일 파라미터 값(중앙값 9자)은 공격(75~600자)과 붙으면 *"짧으면 정상"* 이라는 **새 길이 shortcut**을
  만든다(실측). 전체 쿼리스트링(중앙값 71자)은 길이가 공격과 겹쳐 이 문제를 완화한다.
- CSIC 원본 URL 필드에 딸린 ` HTTP/1.1` 프로토콜 꼬리표가 쿼리에 섞여 들어가 **또 다른 인공
  shortcut**이 됐고(진단 상위 n-gram이 `/1.1`을 지목), 이를 정규식으로 제거했다.

**결과 (TF-IDF+LogReg, clean test)**

| Normal 소스 | Accuracy | Macro-F1 | AUC(OvR) |
|---|---|---|---|
| 원본(영어 산문) `payload_4class` | 0.9765 | 0.9775 | 0.9989 |
| CSIC 실트래픽 `payload_4class_csicnorm` | **0.9927** | **0.9945** | **0.9999** |

**결론 — 편향은 Normal 장르 탓이 아니라 "표면 토큰 포화"가 본질이다**
- Normal 을 실트래픽으로 바꿔도 점수가 **안 떨어지고 오히려 소폭 상승**했다.
  즉 7절 3번(산문 shortcut)은 **주 원인이 아니었다**. 진짜 원인은 7절 2번 **"과제가 쉬움
  (표면 토큰 직교)"** 이다. 공격 3종은 고유 구두점/키워드(`) ( =` / `find | $` / `%3 < </`)로
  이미 완벽히 갈리고, 어떤 Normal 을 넣어도 그 Normal 만의 지문 토큰이 생긴다
  (교체 후엔 CSIC 앱 고유 파라미터 `&B1=`, `nombre=` 등이 그 역할을 했다).
- **함의**: clean·in-distribution 지표는 **데이터를 어떻게 손봐도 포화**된다. 따라서 이 과제에서
  clean 점수로 모델 우열을 가리는 것은 신뢰도가 낮으며, **의미 있는 평가는 RQ2(회피 공격)뿐**이다.
  → 이 실험 자체가 RQ2 의 정당성을 실증한다.

**그럼에도 `csic_normal` 교체를 유지하는 이유**
- clean 점수는 그대로여도, "Normal 이 영화 리뷰"라는 **명백한 방법론적 약점(리뷰어가 즉시 지적할)**
  이 사라진다. Normal 이 실제 HTTP 트래픽이므로 RQ2 의 "benign-evasion(공격→Normal)" 정의도
  현실성을 얻는다. → **RQ2 는 이 `payload_4class_csicnorm` 트랙 위에서 수행**한다.
- 남은 한계(둘 다 논문에 명시): ① CSIC 는 단일 앱(tienda1) 트래픽이라 앱 고유 파라미터가 지문이 됨,
  ② 공격 클래스에 잔존하는 degenerate 샘플(`llll…`, 랜덤 노이즈 소수), ③ 공격 길이 편중.

## 8. 탐지 상보성 분석 — "제안 CNN 이 남들이 놓친 공격을 잡는가?" (실측, 2026-07)

**동기**: Macro-F1 같은 집계 지표는 "전체 성적"만 보여줄 뿐, *"이미지 CNN 이 텍스트 모델이 놓친
공격을 잡는가"* 라는 논문의 핵심 주장에는 답하지 못한다(전체 정확도가 비슷해도 잡는 대상이 다를
수 있다). 이를 샘플 단위로 검증하려고 `src/eval/detection_analysis.py` 를 만들었다.
재현: `python src/eval/detection_analysis.py --track payload_4class_csicnorm --bal --ref cnn --baselines tfidf_logreg,tfidf_rf,charcnn,bilstm`

- **'탐지 성공' 정의**(`metrics.attack_focused` 와 동일): 진짜 공격 샘플에 대해 예측이 Normal 이
  아니면 탐지, Normal 로 새면 미탐(benign-evasion). WAF 관점 실제 위험과 정확히 대응한다.
- 대상: `payload_4class_csicnorm` test 의 **진짜 공격 21,907 건**. 기준(ref)=제안 CNN.

**결과 — 상보성 표** (지표 원본: `experiments/results/detection_complementarity_..._cnn.json`)

| 베이스라인 | 둘 다 탐지 | CNN만 탐지 | 베이스라인만 탐지 | 둘 다 놓침 | CNN recall | 베이스 recall | 순증(CNN) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF+LogReg | 21,885 | **0** | 22 | 0 | 0.9990 | **1.0000** | **−22** |
| TF-IDF+RF | 21,885 | **0** | 22 | 0 | 0.9990 | **1.0000** | **−22** |
| char-CNN | 21,877 | 8 | 22 | 0 | 0.9990 | 0.9996 | **−14** |
| BiLSTM | 21,868 | 17 | 22 | 0 | 0.9990 | 0.9992 | **−5** |

- `CNN만 탐지` = 베이스라인이 Normal 로 흘린 공격을 CNN 이 잡은 수(주장의 직접 근거).
- `베이스라인만 탐지` = 반대 방향(과장 방지용 대칭 지표). 케이스 CSV: `experiments/results/detection_cases/`.

**결론 — 주장은 성립하지 않는다(상보적이되 CNN 열위)**
- **순증이 전부 음수.** 특히 TF-IDF 두 모델은 공격 21,907 건을 **100% 탐지**(recall 1.0)하며,
  CNN 이 이들보다 더 잡는 건은 **0 건**이다. CNN 은 오히려 이들이 잡은 22 건을 놓친다.
- CNN recall(0.9990)은 5 종 중 **최저** — 5절 clean Macro-F1 최하위 결론과 일관.
- **질적으로 더 뼈아픈 점**: CNN 이 놓친 22 건(`cnn_vs_tfidf_logreg_baseONLY.csv`)에는 **명백한
  공격**이 다수다 — `<script>alert(123)</script>`, `<isindex … onbeforeactivate=alert(1)>`,
  `rmdir --ignore-fail-on-non-empty …`, `rsync … user@remote.host`. 라벨 분포는 XSS 17 · CmdInj 4 · SQLi 1.
  반대로 CNN 만 잡은 소수(char-CNN 8·BiLSTM 17)는 대부분 정상과 구분 힘든 짧은 페이로드(`id=55`,
  `or`, `cd`, `j=0`)라 "실력"보다 경계 잡음에 가깝다.
- **함의**: 이미지화 표현은 텍스트 베이스라인 대비 **탐지 상보성에서 이득이 없다**(clean 기준).
  → 제안 모델의 존재 이유는 clean 탐지 우월성이 아니라 **RQ2(회피 강건성)** 에서 찾아야 한다는
  5·7절 결론을 재확인한다.
- ⚠️ 한계: 이 비교는 **clean test 한정**이다. TF-IDF 의 완벽한 clean recall 은 표면 토큰 포화(7절)의
  또 다른 증상이며, 회피 변형 하에서도 유지되는지는 RQ2(05 문서)에서 별도로 다룬다.
