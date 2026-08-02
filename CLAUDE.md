# 프로젝트: 웹 공격 페이로드 이미지화 CNN 탐지 (졸업논문)

## 워크플로우
- Phase 단위 진행. 각 Phase 의 결정/실측/리스크는 `docs/0N_*_design.md`(설계·의사결정 기록)에 남긴다.
- 실측 지표 → `experiments/results/*.json`(⚠️ 전체 .gitignore, 재생성으로 확보 — 커밋 대상 아님), 그림 → `docs/figures/`(추적·커밋 대상).
- 커밋/푸시 대상 브랜치는 `phase1-data-acquisition`(전 Phase 가 여기 쌓임, main 아님).
- git 커밋 메시지(한글 여러 줄)는 파일로 써서 `git commit -F <file>` 사용. Bash 도구에서 PowerShell here-string(`@'...'@`)은 메시지가 깨짐.
- ⚠️ 노트북·데스크톱 양쪽에서 작업 → **착수 전 `git fetch` 필수**. 원격에 Phase 가 쌓여 있으면 문서 번호·모델 이름이 이미 선점됨(실제 사고: `docs/08` 과 모델명 `hybrid` 충돌 → push 거부 후 리베이스·개명 재작업).
- 커밋 전 `python -m pytest tests/ -q` 실행(~3초, GPU 불필요).

## 데이터 파이프라인
- `data/processed`, `data/images` 는 .gitignore(대용량). 재생성으로 확보:
  - `python src/data/preprocess.py`  (seed=42 결정론, CSV 3분할 생성)
  - `python src/imaging/build_image_dataset.py --track <트랙>`  (.npz 생성)
- 트랙: `payload_4class`(RQ1 주), `payload_4class_csicnorm`(Normal=CSIC 실트래픽, RQ2용), `csic_binary`.
- RGB(교수 요구): `build_image_dataset.py --channels rgb [--rgb-encoders raw_byte,char_class,local_entropy]`.
  채널 인코더는 `src/imaging/channel_encoders.py` 레지스트리(교체 가능 — G/B ablation용). npz 는 `_rgb` 접미사, CNN in_channels 자동 추론.
- 흐름 트랙 `ustc_flow_binary`(RQ4b, 악성/정상 이진): `python src/data/download_ustc.py`(USTC-TFC2016, dpkt+py7zr 필요)
  → `python src/imaging/build_flow_dataset.py [--channels rgb]`. flow_to_image 가 세션→앞2304B→48x48, IP/MAC 무력화.
- ⚠️ `preprocess.py <단일트랙>` 은 `docs/03_preprocessing_notes.md` 를 그 트랙만으로 덮어씀 → 노트 유지하려면 인자 없이 전체 실행.
- ⚠️ 새 트랙 추가 시 `--track` choices 를 5곳에서 함께 갱신: preprocess.py(TRACKS/REQUIRED_FILES), build_image_dataset.py, baseline_tfidf.py, diagnose_payload_bias.py, train.py.
- ⚠️ 비-CSV 트랙(예: `ustc_flow_binary`)은 preprocess/build_image_dataset(CSV 경로)를 안 거침 → `--track` choices 를 train.py 한 곳만 갱신(위 "5곳"은 CSV 트랙 한정).
- ⚠️ 새 `data/raw/<dataset>/` 는 자동 무시 안 됨 → `.gitignore` 에 수동 추가(대용량 커밋 사고 방지). 재현은 다운로드 스크립트로 보장.
- ⚠️ `.gitignore` 는 **인라인 주석 불가**(줄 전체가 패턴). `experiments/checkpoints/  # 주석` 형태라 규칙이 무효였고 `.pt` 가 노출돼 있었음(2026-07 수정). 규칙 추가 후 `git check-ignore -v <경로>` 로 반드시 확인.
- ⚠️ RGB npz 파일명 약어 `_ENCODER_ABBR` 는 build_image_dataset.py 와 data_image.py 두 곳에 중복 → 인코더 추가 시 동시 갱신(저장/로드 파일명 일치).

## 모델 / 학습
- 이미지 모델(같은 npz 입력): `cnn`(제안, 93,988p) · `vit`(단일 ViT, 2.7M) · `hybrid`(CNN stem+Transformer, 0.58M) — `src/models/cnn.py`, `vit.py`.
  텍스트 모델: `charcnn` · `bilstm`(`text_models.py`), 별도 sklearn 베이스라인 `baseline_tfidf.py`.
- ⚠️ 새 이미지 모델 추가 시 `IMAGE_MODELS` 를 **2곳** 갱신: `train.py`, `cross_validate.py`(트랙의 "5곳"과는 별개).
- ⚠️ `train.py` 를 `--smoke` 없이 돌리면 **추적 대상 그림**(`docs/figures/models/cm_*.png`)을 덮어씀 → 로컬 코드 검증은 반드시 `--smoke`(저장 생략).
- 체크포인트가 필요한 스크립트(cascade 등)의 로컬 스모크: 스크래치에서 소량 학습한 임시 `.pt` 를 `experiments/checkpoints/` 에 만들고 검증 후 삭제.
- ⚠️ 산출물 tag 는 **실험을 가르는 하이퍼파라미터를 전부 반영**해야 함. 안 그러면 서로 덮어씀(커밋 5eede2f 사고: RGB 조합이 전부 `_rgb` 로 저장돼 상호 덮어쓰기).
  현행 규칙: 채널 `_rgb-rb-cc-bd` (`data_image._channel_suffix`) + ViT 패치 `_p1x48` + 학습률 `_lr0.0003` + 균형화 `_bal`.
  `_lr` 은 기본값(`train.DEFAULT_LR`=1e-3)이면 생략 → 기존 산출물과 파일명 호환. 규칙은 train.py·cross_validate.py **2곳** 동시 갱신.
- ⚠️ **기본 lr=1e-3 은 CNN 기준값** — 단일 ViT 는 이 값에서 발산함(val Macro-F1 이 에폭 간 0.18 폭락).
  lr 만 3e-4/1e-4 로 낮추면 MCC 가 최대 +23.9pp 이동(docs/08 §9.1 실측). ViT 계열 비교는 반드시 lr 을 맞춰서 할 것.
  → "같은 학습 루프를 썼으니 공정"은 성립하지 않음. 기본값이 한쪽 아키텍처에 맞춰져 있으면 나머지가 자동으로 불리해짐.
- 통계 검증은 `src/eval/cross_validate.py`(5-fold, train/val/test 를 풀로 합쳐 재분할). fold 배정이 (라벨, seed)에만 의존 → 설정 간 paired 비교 성립, `label_fingerprint` 로 정렬 검증.
  ⚠️ 단일 split 은 실행 간 ±0.11pp 흔들림(cuDNN 비결정성) → 조합 우열 주장은 반드시 CV 로 판정.

## 실행 환경 (Windows)
- 한글 콘솔(cp949)에서 파이썬 비-ASCII 출력이 깨짐 → Bash 로 파이썬 실행 시 `PYTHONIOENCODING=utf-8` 프리픽스 사용(또는 스크립트가 stdout 재설정).
- matplotlib 그림 라벨/범례/제목은 ASCII 만(DejaVu Sans 에 Hangul 없음 → □ 로 깨짐+경고). 한글은 콘솔·JSON 에만.
- **GPU 는 데스크톱 한정**(RTX 4080 SUPER, 17GB / torch cu124, RAM 34GB) — 전면 학습은 거기서 실행.
  노트북은 Intel Arc iGPU + torch CPU 빌드라 학습 불가 → 착수 시 `torch.cuda.is_available()` 로 어느 머신인지 먼저 확인.
  ⚠️ GPU 는 1대뿐 → 학습 작업은 **순차 실행**(동시 실행 시 재현성 악화). 학습 중 다른 검증이 필요하면 `CUDA_VISIBLE_DEVICES=` 로 CPU 강제.
- ⚠️ 백그라운드 파이썬은 stdout 버퍼링으로 진행 로그가 안 보임 → 진행 추적은 산출 JSON 존재 여부로 하거나 `python -u` 사용.
- ⚠️ **백그라운드 작업이 "killed" 로 보고돼도 파이썬 자식 프로세스는 살아 있을 수 있음**(셸 래퍼만 종료됨).
  로그·산출물만 보고 "죽었다" 판단하면 재실행 시 GPU 동시 실행이 발생해 순차 실행 규칙이 깨짐(2026-07-19 실제 사고, docs/08 §9.6).
  → 재실행 전 반드시 생존 확인: `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` 또는 `nvidia-smi` 로 GPU 점유 확인.
- 장시간 작업(CV 등)은 완료 감지를 `until [ -f <산출 JSON> ]; do sleep 20; done` 로 걸 것(래퍼 종료와 무관하게 동작).
- ⚠️ Bash 도구에서 `git show <ref>:<path>` 는 `:` 가 `;` 로 변환돼 실패 → `git diff <a> <b> -- <path>` 사용.
- `nvidia-smi` 는 PATH 에 없음 → GPU 확인은 `torch.cuda.is_available()` 또는 PowerShell `Get-CimInstance Win32_VideoController`.
- 클래스 불균형 트랙 학습: `train.py --balance`(train 만 언더샘플링, val/test 실분포 유지).
- 모든 모델 평가는 `src/eval/metrics.py` 로 일원화(계산 방식 차이 배제). 지표: Acc·Macro-F1·MCC·PR-AUC·ROC-AUC + attack_focused(benign-evasion/FPR). 불균형 보안 데이터 헤드라인은 MCC·PR-AUC·benign-evasion(문헌 근거 docs/07).

## 모델 용어 (2026-08-02 변경 — 혼동 사고 후 확정)
- **제안 모델 = 캐스케이드**(1차 RGB CNN + 2차 char-CNN). 논문이 주장하는 대상은 이것.
- ⚠️ 단일 CNN 은 제안 지위를 내려놓고 **지표 비교 기준**이 됐다. `제안 CNN` 이라는 이름은 **쓰지 말 것** — 그 말이 (ㄱ)제안 모델과 (ㄴ)캐스케이드 1차 부품 두 뜻으로 읽혀 실제로 혼동이 났다(docs/10 §0.1).
  - `RGB CNN` = 48×48×3 단일 CNN(기본 비교 기준) · `gray CNN` = 1채널 판본 · `얕은 CNN` = 채널 무관 아키텍처 지칭.
  - ⚠️ 과거 문서의 회피 수치 0.594/0.5944/0.0022 는 **gray 실측**이다. RGB 기준선은 **0.5981**(2026-08-02 측정).

## 캐스케이드 (Phase 10, docs/09 · Phase 11 확장 docs/10 §5.1)
- ⚠️ **`hybrid`(모델, docs/08 ViT 계열 = CNN stem+Transformer)와 `cascade`(배치 구조)는 다른 것** — 이름 섞지 말 것.
- `src/models/cascade.py`: 1차=RGB CNN(전량) → 확신도<τ 인 소수만 2차=char-CNN. **학습 없음**(기존 `_bal` 체크포인트 재사용). τ→0 은 CNN 단독, τ→1 은 char-CNN 단독으로 정확히 수렴.
- τ 는 **val 에서 확정 후 test 에 1회 적용**(test 로 고르면 누수). `--select match-teacher|budget`.
- 회피 실험 편입: `run_evasion.py --model cascade` — τ 는 산출 JSON 에서 읽음(공격 데이터로 재튜닝 금지).
- 방어 구성(Phase 11): `cascade.py --defense advtrain --mutation-split {S0,SA} --aug-ratio ρ` → **1·2차 양쪽** 방어본을 조립하고 τ 를 그 구성의 val 에서 **재선택**한다(확신도 분포가 달라 기존 τ 재사용 불가).
- 체크포인트/산출물 tag 는 `tagging.build_tag` 단일 진실 소스 + 캐스케이드 고유 축(2차 모델·τ 선택 규칙·방어) 추가. ⚠️ cascade.py 가 규칙 사본을 들고 있었으나 tagging.py 로 통합됨(2026-08-02).
- ⚠️ **"막아둔 조합"이 tag 버그를 가린다**: `run_evasion` 이 캐스케이드+채널을 `p.error` 로 막고 있던 동안, 저장 tag 의 채널 축이 `model=="cnn"` 조건에 묶여 있는 걸 아무도 못 봤다. 차단을 푼 즉시 RGB 결과가 gray 결과를 덮어씀(2026-08-02, docs/10 §5.1). → **축은 그 조합을 쓸 수 있게 되기 전에 tag 에 넣어둘 것.** 차단을 풀 때는 저장 경로의 tag 부터 점검.
- ⚠️ 캐스케이드가 줄이는 건 **추론 지연**뿐. 두 모델을 다 학습하므로 **학습 비용은 합산** — 근거는 처리량(docs/04 §5: 81,964 vs 6,010 /s)이지 학습 지표(s/epoch)가 아님.
- ⚠️ 캐스케이드 고유 리스크: 회피 변형이 1차 확신도를 낮추면 에스컬레이션↑ → **정확도 그대로인데 지연만 폭증**(비용 기반 공격). `run_evasion` 이 조건별 `escalation_rate` 를 함께 기록.
