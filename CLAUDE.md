# 프로젝트: 웹 공격 페이로드 이미지화 CNN 탐지 (졸업논문)

## 워크플로우
- Phase 단위 진행. 각 Phase 의 결정/실측/리스크는 `docs/0N_*_design.md`(설계·의사결정 기록)에 남긴다.
- 실측 지표 → `experiments/results/*.json`(⚠️ 전체 .gitignore, 재생성으로 확보 — 커밋 대상 아님), 그림 → `docs/figures/`(추적·커밋 대상).
- 커밋/푸시 대상 브랜치는 `phase1-data-acquisition`(전 Phase 가 여기 쌓임, main 아님).
- git 커밋 메시지(한글 여러 줄)는 파일로 써서 `git commit -F <file>` 사용. Bash 도구에서 PowerShell here-string(`@'...'@`)은 메시지가 깨짐.

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
- ⚠️ RGB npz 파일명 약어 `_ENCODER_ABBR` 는 build_image_dataset.py 와 data_image.py 두 곳에 중복 → 인코더 추가 시 동시 갱신(저장/로드 파일명 일치).

## 실행 환경 (Windows)
- 한글 콘솔(cp949)에서 파이썬 비-ASCII 출력이 깨짐 → Bash 로 파이썬 실행 시 `PYTHONIOENCODING=utf-8` 프리픽스 사용(또는 스크립트가 stdout 재설정).
- matplotlib 그림 라벨/범례/제목은 ASCII 만(DejaVu Sans 에 Hangul 없음 → □ 로 깨짐+경고). 한글은 콘솔·JSON 에만.
- torch 학습은 GPU 필요(외부 실행). 로컬 CPU 는 `--smoke` 또는 sklearn 베이스라인(`baseline_tfidf.py`)만 실행.
- 클래스 불균형 트랙 학습: `train.py --balance`(train 만 언더샘플링, val/test 실분포 유지).
- 모든 모델 평가는 `src/eval/metrics.py` 로 일원화(계산 방식 차이 배제). 지표: Acc·Macro-F1·MCC·PR-AUC·ROC-AUC + attack_focused(benign-evasion/FPR). 불균형 보안 데이터 헤드라인은 MCC·PR-AUC·benign-evasion(문헌 근거 docs/07).
