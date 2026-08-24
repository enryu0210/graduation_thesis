# 웹 공격 페이로드 이미지화 기반 캐스케이드 탐지 및 적대적 강건성 연구

학부 졸업논문 프로젝트. 웹 공격 페이로드(SQLi/XSS/Command Injection)를 **바이트 이미지(48×48 RGB)**
로 변환해 탐지하고, 그 위에서 **회피 공격·방어·비용 기반 공격**까지 다룬다.

**제안 모델은 캐스케이드다** — 1차 RGB CNN 이 전량을 보고, 확신도가 임계값 τ 미만인 소수만
2차 char-CNN 으로 넘긴다(추가 학습 없음). 이미지 단독 CNN 이 텍스트 모델에 **유의하게 열세**라는
음성 결과에서 출발한 설계이며, 그 음성 결과는 숨기지 않고 논문에 그대로 보고한다.

> 📌 **처음 읽는다면 [`docs/CURRENT_BASELINE.md`](docs/CURRENT_BASELINE.md) 부터.**
> RQ·평가지표·데이터셋이 시기별로 어떻게 바뀌었고 **지금 무엇이 유효한지**를 한 문서에 모아 뒀다.
> 옛 문서의 MCC 수치나 `제안 CNN` 같은 표기를 만났을 때의 해석 규칙도 거기 있다.

## 문서 지도

| 문서 | 무엇 |
|---|---|
| [`docs/CURRENT_BASELINE.md`](docs/CURRENT_BASELINE.md) | **현행 기준 통합 참조본** — 시기 구분, 유효/폐기 대조, 자주 헷갈리는 것 |
| [`docs/web_attack_image_cnn_thesis_design.md`](docs/web_attack_image_cnn_thesis_design.md) | **마스터 설계** — RQ 표·기여 문구·논문 목차의 단일 진실 소스 |
| [`docs/13_dataset_metric_revision_design.md`](docs/13_dataset_metric_revision_design.md) | **지표·데이터셋 재확정**(2026-08-13~) — 이 두 축은 마스터보다 이쪽이 최신 |
| [`docs/DATA_MANIFEST.md`](docs/DATA_MANIFEST.md) | 데이터 출처·라이선스·체크섬·실제 행 수 (재현성 근거) |
| `docs/0N_*.md` | Phase 별 설계·의사결정 기록 (01 데이터확보 · 03 전처리/이미지화 · 04 모델/RQ1 · 05 회피/RQ2 · 06 암호화경계/RQ4 · 07 관련연구·RGB · 08 ViT · 09 캐스케이드 · 10 방어/RQ3 · 11 캐스케이드 확장 · 12 비용축 선행대조 · 13 지표·데이터셋) |
| [`docs/reports/`](docs/reports/) | 저장소 밖으로 나가는 보고서(교수 제출용 PDF·DOCX) |
| [`docs/archive/`](docs/archive/) | 지나간 시점의 기록 — ⚠️ 현행 기준이 아님 |
| [`CLAUDE.md`](CLAUDE.md) | 코드 관례·함정(트랙 추가 "5곳", tag 규칙, GPU 순차 실행 등) |

## 연구 질문

| ID | 질문 | 상태 |
|---|---|---|
| RQ1 | 이미지화 탐지는 텍스트 모델 대비 **정확도–비용 Pareto 상 어디에 있는가** | ✅ 답 완료 — **음성 결과**(이미지 단독은 진다) |
| RQ5 ⭐ | 캐스케이드는 그 두 점을 τ 하나로 잇는 **연속 곡선**으로 바꾸는가 | ✅ 같은 정확도를 **5.07배 싸게** |
| RQ2 | 캐스케이드는 WAF 우회 기법에 얼마나 취약한가 + **비용 기반 공격**이 성립하는가 | ✅ 답 완료 |
| RQ3 | 적대적 증강은 **미지 변형까지 일반화**되는가, 대가는 어느 통화로 지불되는가 | ✅ 답 완료 |
| RQ4 | 인코딩→암호화로 신호가 사라질 때 내용 기반 탐지는 **어디서 붕괴하는가** | ✅ RQ4a / ⚠️ RQ4b 포화 |
| RQ6 | 1·2차의 **표현 불일치**가 회피 시도의 탐지 신호가 되는가 | ⚠️ 부분 답(보조 지표) |

⚠️ **RQ 번호는 재배열하지 않는다** — 파일명에 박혀 있다. 새 질문은 뒤 번호로 추가한다.

## 디렉토리 구조

```
data/raw/         # 원본 데이터 (손대지 않음). 대용량은 .gitignore 로 제외
data/processed/   # 정제·분할 완료된 CSV (재생성으로 확보)
data/images/      # 바이트 -> 이미지 변환 결과 .npz (재생성으로 확보)
src/data/         # 다운로드·EDA·전처리·검증
src/imaging/      # 페이로드/흐름 -> 이미지 변환, 채널 인코더 레지스트리
src/models/       # CNN·ViT·hybrid·char-CNN·BiLSTM·TF-IDF + 학습 스크립트 + 캐스케이드
src/attacks/      # RQ2 회피 공격 (problem-space 변형)
src/defense/      # RQ3 적대적 증강
src/eval/         # 지표 일원화·5-fold CV·유의성 검정
experiments/      # 체크포인트(.pt)·결과 JSON — 전부 .gitignore, 재생성으로 확보
docs/figures/     # 그림 (추적·커밋 대상)
```

## 진행 상황

| Phase | 작업 | 상태 |
|---|---|---|
| 1–3 | 데이터 확보·EDA·전처리·이미지 변환 | ✅ |
| 4 | RQ1 모델 비교 (이미지 vs 텍스트) | ✅ 음성 결과 확정 |
| 5 | RQ2 회피 공격 | ✅ |
| 6 | RQ4 암호화 경계 | ✅ RQ4a / ⚠️ RQ4b 포화 |
| 7 | 관련연구·지표 개편·RGB 채널 | ✅ |
| 8–9 | ViT / hybrid arm | ✅ **미채택 확정** |
| 10 | **캐스케이드** (제안 모델로 전환) | ✅ GPU 실측 완료 |
| 11 | RQ3 적대적 방어 | ✅ |
| 12 | 캐스케이드 확장 M1·M2·M4 + 비용 축 선행 대조 | ✅ M2 조건부 채택 / M1 하향 / M4 미채택 |
| **13** | **데이터셋·평가지표 재확정** | 🔄 지표 개편 완료 · **데이터셋 교체 미착수** |

⚠️ **현재 `experiments/results/` 의 수치는 전부 데이터셋 교체 이전 기준**이며 재측정 대상이다.
살아남는 것은 코드·아이디어·방법론 전부다 — 자세한 범위는 docs/13 §4.

## 설치 및 사용

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # ~5초, GPU 불필요. 커밋 전 실행
```

### 데이터 준비 (재생성)

```bash
python src/data/validate.py                    # data/raw 통계 -> docs/DATA_MANIFEST.md 갱신
python src/data/download_csic2010.py           # 사전에 ~/.kaggle/kaggle.json 필요
python src/data/preprocess.py                  # seed=42 결정론, CSV 3분할 생성
python src/imaging/build_image_dataset.py --track payload_4class_csicnorm --channels rgb
```

⚠️ `preprocess.py <단일트랙>` 은 `docs/03_preprocessing_notes.md` 를 그 트랙만으로 덮어쓴다 →
노트를 유지하려면 **인자 없이 전체 실행**한다.

### 학습·평가

```bash
# 이미지 모델 (cnn · cnn_ee · vit · hybrid) — 같은 .npz 입력
python src/models/train.py --model cnn --track payload_4class_csicnorm --channels rgb --balance

# 텍스트 모델
python src/models/train.py --model charcnn --track payload_4class_csicnorm --balance
python src/models/baseline_tfidf.py --track payload_4class_csicnorm --clf logreg

# 코드 점검용 스모크 (그림 저장 생략)
python src/models/train.py --model cnn --smoke
```

⚠️ `--smoke` 없이 돌리면 **추적 대상 그림**(`docs/figures/models/cm_*.png`)을 덮어쓴다.
로컬 코드 검증은 반드시 `--smoke` 로 한다.
⚠️ 기본 `lr=1e-3` 은 **CNN 기준값**이다. ViT 계열은 이 값에서 발산하므로 비교 시 lr 을 맞춘다.

### 캐스케이드 (제안 모델) · 회피 · 통계 검증

```bash
# 기존 체크포인트 2개를 조합 (학습 없음). τ 는 val 에서 확정 후 test 에 1회 적용
# ⚠️ 체크포인트를 찾으려면 학습 때와 같은 축(--channels·--balance)을 그대로 넘겨야 한다
python src/models/cascade.py --track payload_4class_csicnorm --channels rgb --balance --select match-teacher

# 회피 공격 (τ 는 산출 JSON 에서 읽음 — 공격 데이터로 재튜닝하지 않는다)
python src/attacks/run_evasion.py --model cascade --track payload_4class_csicnorm --channels rgb

# 5-fold CV + paired t-test + Holm 보정 (조합 우열은 반드시 CV 로 판정)
python src/eval/cross_validate.py --model cnn --track payload_4class_csicnorm --channels rgb
```

지표는 `experiments/results/*.json`, 그림은 `docs/figures/` 에 저장된다.
모든 모델의 지표는 `src/eval/metrics.py` 한 곳으로 일원화돼 있다(계산 방식 차이 배제).

## 실행 환경 주의

- **GPU 는 데스크톱 한정**(RTX 4080 SUPER / torch cu124). 노트북은 CPU 빌드라 전면 학습 불가 →
  착수 시 `torch.cuda.is_available()` 로 어느 기기인지 먼저 확인한다.
- **GPU 는 1대뿐** → 학습 작업은 **순차 실행**. 동시 실행하면 재현성이 나빠진다.
- 한글 콘솔(cp949)에서 비-ASCII 출력이 깨진다 → `PYTHONIOENCODING=utf-8` 프리픽스 사용.
- matplotlib 그림의 라벨·범례·제목은 **ASCII 만**(DejaVu Sans 에 한글 없음).
- 커밋·푸시 대상 브랜치는 `phase1-data-acquisition`(main 아님).
