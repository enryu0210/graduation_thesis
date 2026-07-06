# 웹 공격 페이로드 이미지화 기반 CNN 탐지 및 적대적 강건성 연구

학부 졸업논문 프로젝트. 웹 공격 페이로드(SQLi/XSS/Command Injection)를 바이트 단위
그레이스케일 이미지로 변환해 CNN 으로 탐지하고(RQ1), WAF 우회 회피 공격에 대한
취약성(RQ2)과 adversarial training 을 통한 강건성 개선(RQ3)을 연구한다.

- 전체 설계: [`docs/web_attack_image_cnn_thesis_design.md`](docs/web_attack_image_cnn_thesis_design.md)
- Phase 1 데이터 확보 계획: [`docs/01_data_acquisition_plan.md`](docs/01_data_acquisition_plan.md)
- 데이터 출처/체크섬 기록: [`docs/DATA_MANIFEST.md`](docs/DATA_MANIFEST.md)
- Phase 3 전처리·이미지화 설계: [`docs/03_preprocessing_imaging_design.md`](docs/03_preprocessing_imaging_design.md)
- Phase 4 모델·RQ1 설계: [`docs/04_models_rq1_design.md`](docs/04_models_rq1_design.md)
- Phase 5 회피 공격·RQ2 설계: [`docs/05_evasion_attacks_rq2_design.md`](docs/05_evasion_attacks_rq2_design.md)

## 디렉토리 구조

```
data/raw/         # 원본 데이터 (손대지 않음). 대용량은 .gitignore 로 제외.
data/interim/     # 디코딩·정규화 등 중간 산출물
data/processed/   # 정제·분할 완료된 최종 테이블
data/images/      # (Phase 3) 바이트 -> 이미지 변환 결과
src/data/         # 다운로드·검증 스크립트
src/imaging/      # (Phase 3) 페이로드 -> 이미지 변환
src/models/       # (Phase 4) 제안 CNN / 베이스라인(TF-IDF, char-CNN, BiLSTM) + 학습 스크립트
src/attacks/      # (Phase 5) RQ2 회피 공격
src/defense/      # (Phase 6) RQ3 adversarial training
src/eval/         # 지표 계산·유의성 검정
```

## 현재 진행 상황

- [x] Phase 1 — 데이터 확보/검증 (payload_4class 199,793행 + CSIC 2010)
- [x] Phase 2 — EDA (페이로드 길이 분포 → 이미지 폭 W=48 결정)
- [x] Phase 3 — 전처리·분할 + 바이트→이미지(.npz) 변환
- [x] Phase 4 — RQ1 실험 완료: 5개 모델 GPU 전면 학습·평가 (제안 CNN Macro-F1 0.950 vs 텍스트 베이스라인 0.977~0.996)
- [x] Phase 4.5 — Normal 편향 진단: CSIC 실트래픽으로 교체해도 clean 포화(0.99+) 유지 → 표면토큰이 본질임을 실증(04문서 §7.1). RQ2용 `payload_4class_csicnorm` 트랙 확보
- [~] Phase 5 — 회피 공격(RQ2): **설계 확정**([`docs/05_...`](docs/05_evasion_attacks_rq2_design.md)), 구현 착수 전
- [ ] Phase 6 — adversarial training(RQ3)

> 상세·열린 결정은 각 Phase 설계 문서 참조. Phase 4 실제 학습은 GPU(Colab/데스크톱) 권장.

## 설치 및 사용

```bash
pip install -r requirements.txt

# 데이터 검증 (data/raw 전체 통계 -> docs/DATA_MANIFEST.md 갱신)
python src/data/validate.py

# CSIC 2010 다운로드 (사전에 ~/.kaggle/kaggle.json 필요)
python src/data/download_csic2010.py

# 페이로드 보강 소스 clone
python src/data/download_supplements.py
```

### Phase 4 — 모델 학습·평가 (RQ1)

```bash
# 베이스라인 ① TF-IDF + 전통 ML (sklearn 만 필요, GPU 불필요)
python src/models/baseline_tfidf.py --track payload_4class --clf logreg

# 제안 CNN + 텍스트 베이스라인 (torch 필요, GPU 권장)
python src/models/train.py --model cnn        # 제안 모델
python src/models/train.py --model charcnn    # char-CNN
python src/models/train.py --model bilstm     # BiLSTM

# 코드 점검용 스모크(작게 실행)
python src/models/train.py --model cnn --smoke
```

지표는 `experiments/results/*.json`, 혼동행렬은 `docs/figures/models/` 에 저장됩니다.
자세한 설계·실행(Colab)은 [`docs/04_models_rq1_design.md`](docs/04_models_rq1_design.md) 참조.
