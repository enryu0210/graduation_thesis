# 웹 공격 페이로드 이미지화 기반 CNN 탐지 및 적대적 강건성 연구

학부 졸업논문 프로젝트. 웹 공격 페이로드(SQLi/XSS/Command Injection)를 바이트 단위
그레이스케일 이미지로 변환해 CNN 으로 탐지하고(RQ1), WAF 우회 회피 공격에 대한
취약성(RQ2)과 adversarial training 을 통한 강건성 개선(RQ3)을 연구한다.

- 전체 설계: [`docs/web_attack_image_cnn_thesis_design.md`](docs/web_attack_image_cnn_thesis_design.md)
- Phase 1 데이터 확보 계획: [`docs/01_data_acquisition_plan.md`](docs/01_data_acquisition_plan.md)
- 데이터 출처/체크섬 기록: [`docs/DATA_MANIFEST.md`](docs/DATA_MANIFEST.md)

## 디렉토리 구조

```
data/raw/         # 원본 데이터 (손대지 않음). 대용량은 .gitignore 로 제외.
data/interim/     # 디코딩·정규화 등 중간 산출물
data/processed/   # 정제·분할 완료된 최종 테이블
data/images/      # (Phase 3) 바이트 -> 이미지 변환 결과
src/data/         # 다운로드·검증 스크립트
src/imaging/      # (Phase 3) 페이로드 -> 이미지 변환
src/models/       # (Phase 4) CNN / 베이스라인
src/attacks/      # (Phase 5) RQ2 회피 공격
src/defense/      # (Phase 6) RQ3 adversarial training
src/eval/         # 지표 계산·유의성 검정
```

## 현재 진행 상황 (Phase 1)

- [x] 프로젝트 스켈레톤 생성
- [x] 3-class 페이로드 데이터셋 확보 및 1차 검증 (206,636행, 4-class)
- [ ] CSIC 2010 다운로드 (Kaggle 인증 필요)
- [ ] 페이로드 보강 소스 clone
- [ ] 데이터셋 정확한 출처/라이선스 확정

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
