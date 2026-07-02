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

## 5. Phase 4 검증 현황 (이 노트북, CPU)

- torch **불필요** 경로는 실데이터로 검증 완료:
  - TF-IDF + LogisticRegression (payload_4class): **acc≈0.976, Macro-F1≈0.977, AUC≈0.999**
- torch 경로(cnn/charcnn/bilstm)는 **스모크 테스트로 무결성만 확인**(작은 배치 학습→평가→저장까지 완주).
  전면 학습은 GPU 환경에서 수행 예정.
- 단위 테스트 `tests/test_models.py` 28건 통과(지표·클래스가중치·시퀀스 인코딩·모델 출력 shape).

---

## 6. 리스크 / 열린 결정 (다음 단계로 넘김)

- [ ] GPU 전면 학습 후 4개 모델 지표 표/그림 확정, 5-fold CV + paired t-test(설계 6.1)
- [ ] ablation: decoded vs raw, side 32/48/64, RGB 강화(설계 4장 Step 3)
- [ ] `csic_binary` 일반화 트랙 평가(이미지화 단위 URL vs 전체요청 재검토 포함)
- [ ] 최종 모델 확정 → Phase 5(RQ2 회피 공격)의 공격 대상 모델로 사용
- [ ] (운영) 브랜치 정리 — 현재 Phase 2~4 가 모두 `phase1-data-acquisition` 브랜치에
  커밋돼 있어 이름과 내용이 불일치. phase 별 브랜치 분리 또는 브랜치명 변경 검토 필요
