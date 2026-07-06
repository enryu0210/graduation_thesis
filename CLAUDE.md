# 프로젝트: 웹 공격 페이로드 이미지화 CNN 탐지 (졸업논문)

## 워크플로우
- Phase 단위 진행. 각 Phase 의 결정/실측/리스크는 `docs/0N_*_design.md`(설계·의사결정 기록)에 남긴다.
- 실측 지표 → `experiments/results/*.json`, 그림 → `docs/figures/`.
- 커밋/푸시 대상 브랜치는 `phase1-data-acquisition`(전 Phase 가 여기 쌓임, main 아님).

## 데이터 파이프라인
- `data/processed`, `data/images` 는 .gitignore(대용량). 재생성으로 확보:
  - `python src/data/preprocess.py`  (seed=42 결정론, CSV 3분할 생성)
  - `python src/imaging/build_image_dataset.py --track <트랙>`  (.npz 생성)
- 트랙: `payload_4class`(RQ1 주), `payload_4class_csicnorm`(Normal=CSIC 실트래픽, RQ2용), `csic_binary`.
- ⚠️ `preprocess.py <단일트랙>` 은 `docs/03_preprocessing_notes.md` 를 그 트랙만으로 덮어씀 → 노트 유지하려면 인자 없이 전체 실행.
- ⚠️ 새 트랙 추가 시 `--track` choices 를 5곳에서 함께 갱신: preprocess.py(TRACKS/REQUIRED_FILES), build_image_dataset.py, baseline_tfidf.py, diagnose_payload_bias.py, train.py.

## 실행 환경 (Windows)
- 한글 콘솔(cp949)에서 파이썬 비-ASCII 출력이 깨짐 → Bash 로 파이썬 실행 시 `PYTHONIOENCODING=utf-8` 프리픽스 사용(또는 스크립트가 stdout 재설정).
- torch 학습은 GPU 필요(외부 실행). 로컬 CPU 는 `--smoke` 또는 sklearn 베이스라인(`baseline_tfidf.py`)만 실행.
- 클래스 불균형 트랙 학습: `train.py --balance`(train 만 언더샘플링, val/test 실분포 유지).
- 모든 모델 평가는 `src/eval/metrics.py` 로 일원화(계산 방식 차이 배제).
