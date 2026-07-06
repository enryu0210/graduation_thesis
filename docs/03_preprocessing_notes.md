# 전처리·분할 결과 (preprocess.py)

> `python src/data/preprocess.py` 실행 시 자동 생성됩니다.
> 설계 3장 전처리 체크리스트(디코딩·중복제거·stratified split) 구현 결과입니다.

- 분할 비율: train 70% / val 15% / test 15% (stratified, seed=42)
- 각 산출 CSV 컬럼: `text_raw`, `text_decoded`, `label` (raw/decoded 를 모두 보존해 디코딩 ablation 지원)

---

## 트랙: payload_4class

- 시작 행 수: 206,636
- 빈 문자열 제거: 616 행 → 남은 206,020
- 완전 중복 제거(raw 기준): 6,227 행 (3.02%) → 남은 199,793
- 분할 결과(행 수 및 클래스 분포):
    - train: 139,855 (70.0%) | CommandInjection=33,610, Normal=37,626, SQLInjection=40,096, XSS=28,523
    - val: 29,969 (15.0%) | CommandInjection=7,202, Normal=8,063, SQLInjection=8,592, XSS=6,112
    - test: 29,969 (15.0%) | CommandInjection=7,202, Normal=8,062, SQLInjection=8,592, XSS=6,113
- 저장 파일:
    - `data\processed\payload_4class_train.csv` (139,855 행)
    - `data\processed\payload_4class_val.csv` (29,969 행)
    - `data\processed\payload_4class_test.csv` (29,969 행)

## 트랙: payload_4class_csicnorm

- 시작 행 수: 156,115
- 빈 문자열 제거: 0 행 → 남은 156,115
- 완전 중복 제거(raw 기준): 5,260 행 (3.37%) → 남은 150,855
- 분할 결과(행 수 및 클래스 분포):
    - train: 105,598 (70.0%) | CommandInjection=33,610, Normal=3,368, SQLInjection=40,096, XSS=28,524
    - val: 22,628 (15.0%) | CommandInjection=7,202, Normal=722, SQLInjection=8,592, XSS=6,112
    - test: 22,629 (15.0%) | CommandInjection=7,202, Normal=722, SQLInjection=8,592, XSS=6,113
- 저장 파일:
    - `data\processed\payload_4class_csicnorm_train.csv` (105,598 행)
    - `data\processed\payload_4class_csicnorm_val.csv` (22,628 행)
    - `data\processed\payload_4class_csicnorm_test.csv` (22,629 행)

## 트랙: csic_binary

- 시작 행 수: 61,065
- 빈 문자열 제거: 0 행 → 남은 61,065
- 완전 중복 제거(raw 기준): 47,567 행 (77.90%) → 남은 13,498
- 분할 결과(행 수 및 클래스 분포):
    - train: 9,448 (70.0%) | Anomalous=6,060, Normal=3,388
    - val: 2,025 (15.0%) | Anomalous=1,299, Normal=726
    - test: 2,025 (15.0%) | Anomalous=1,299, Normal=726
- 저장 파일:
    - `data\processed\csic_binary_train.csv` (9,448 행)
    - `data\processed\csic_binary_val.csv` (2,025 행)
    - `data\processed\csic_binary_test.csv` (2,025 행)

